"""One call from prompt (+ optional pictures) to a four-way seamless image. The method is in torus.py.

    from flux2.torus_generate import TorusPipe, generate
    pipe = TorusPipe("flux.2-klein-base-9b")
    image = generate(pipe, "seamless rose pattern ...", seed=0, ref_image="a.png")[0]     # uint8 [H, W, 3]
    save_png(image, "roses.png", tiled=True)

`generate` takes one prompt or a list of them (with one seed each) and runs them as one batch.
scripts/torus_cli.py is this function plus files: prompt lists, shards, folders, the gallery.
"""

import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from .sampling import (
    batched_prc_img,
    batched_prc_txt,
    cap_min_pixels,
    cap_pixels,
    center_crop_to_multiple_of_x,
    default_images_prep,
    get_schedule,
    prc_img,
)
from .torus import DECODE_PAD_TOKENS, TORUS_PROMPT, build_torus_geometry, decode_torus, denoise_torus
from .util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder


class TorusPipe:
    """The three networks, and a cache of text embeddings so the text encoder can leave the GPU."""

    def __init__(self, model_name: str = "flux.2-klein-base-9b"):
        assert not FLUX2_MODEL_INFO[model_name]["guidance_distilled"], "real CFG needs an undistilled model"
        self.text_encoder = load_text_encoder(model_name, device=torch.device("cuda")).eval()
        self.model = load_flow_model(model_name, device=torch.device("cuda")).eval()
        self.ae = load_ae(model_name).eval()
        self.ctx_of: dict[str, Tensor] = {}  # text -> [512, D] bf16 on the CPU

    @torch.no_grad()
    def encode_text(self, texts: list[str]):
        todo = sorted(set(texts) - set(self.ctx_of))
        for i in range(0, len(todo), 8):
            for t, c in zip(todo[i : i + 8], self.text_encoder(todo[i : i + 8]).to(torch.bfloat16).cpu()):
                self.ctx_of[t] = c

    def drop_text_encoder(self):
        """On cards without FP8 (A40) the 8B encoder is 16 GB of bf16 next to the 18 GB flow model.
        Encode everything a run needs, then call this; `generate` only reads the cache."""
        self.text_encoder = None
        torch.cuda.empty_cache()


def load_pixels(image: str | Image.Image, width: int, height: int) -> Tensor:
    """[1, 3, height, width] in [-1, 1] on the GPU, resized (not cropped) to the generation size."""
    image = Image.open(image) if isinstance(image, str) else image
    image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    return default_images_prep(image)[None].cuda()


def fit_reference(image: str | Image.Image, max_pixels: int = 512**2) -> Image.Image:
    """A reference image the way FLUX.2 wants it (sampling.py: default_prep), with a smaller area cap.

    FLUX.2 does not ask a reference to match the output: any size, any aspect ratio up to 8:1, no
    side under 64 px, area capped (stock: 2024^2 for one reference, 1024^2 each for several), then
    centre-cropped to a multiple of 16. The aspect ratio is kept, never squashed to the output's.
    The cap here is lower because every 16x16 px of reference is one more token in attention that is
    written out by hand: 512^2 adds 1024 tokens, 1024^2 adds 4096.

    Museum scans can be gigapixels; a JPEG is decoded straight at a reduced size (`draft`)."""
    if isinstance(image, str):
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(image)
        w, h = image.size
        shrink = max(1.0, (w * h / max_pixels) ** 0.5)
        image.draft("RGB", (int(w / shrink) + 1, int(h / shrink) + 1))  # JPEG only; a no-op otherwise
    image = cap_pixels(cap_min_pixels(image.convert("RGB")), max_pixels)
    return center_crop_to_multiple_of_x(image, 16)


def save_png(image: Tensor, path: str, tiled: bool = False):
    """`image`: one uint8 [H, W, 3] from `generate`. tiled=True also writes <path>_tiled.png, 2x2 copies."""
    Image.fromarray(image.numpy()).save(path)
    if tiled:
        Image.fromarray(image.repeat(2, 2, 1).numpy()).save(str(path)[:-4] + "_tiled.png")


def keep_grid(
    width: int, height: int, keep_mask: str | Image.Image | None = None, keep_center: float = 0.0
) -> Tensor:
    """[gh, gw] bool on the GPU: the tokens that must stay untouched.
    Mask: white = untouched, and a token is kept only if all of its 16x16 pixels are white.
    Otherwise the centred rectangle covering `keep_center` of each side."""
    gh, gw = height // 16, width // 16
    if keep_mask is not None:
        white = load_pixels(keep_mask, width, height).mean(1, keepdim=True) > 0
        return F.avg_pool2d(white.float(), 16)[0, 0] == 1
    mh, mw = round(gh * (1 - keep_center) / 2), round(gw * (1 - keep_center) / 2)
    grid = torch.zeros(gh, gw, dtype=torch.bool, device="cuda")
    grid[mh : gh - mh, mw : gw - mw] = True
    return grid


@torch.no_grad()
def generate(
    pipe: TorusPipe,
    prompt: str | list[str],
    seed: int | list[int] = 0,  # one per prompt; an image depends on its own seed, not on its batch
    width: int = 256,
    height: int = 256,
    num_steps: int = 50,
    guidance: float = 4.0,
    geo_guidance: float = 6.0,  # untuned (the reference uses 6 for ERP on FLUX.2-dev); == guidance is plain CFG
    geo_prompt: str = TORUS_PROMPT,
    wrap: tuple[bool, bool] = (True, True),  # (h, w)
    unanchor_text: bool = False,
    ref_image: str
    | Image.Image
    | list
    | None = None,  # FLUX.2 reference(s), shared by the batch; "a.png,b.png" ok
    ref_max_pixels: int = 512**2,  # see fit_reference
    init_image: str | Image.Image | None = None,  # the picture whose kept region stays untouched
    keep_mask: str | Image.Image | None = None,
    keep_center: float = 0.0,
    paste_kept_pixels: bool = True,  # kept latents decode to *almost* the original; this makes it exact
    decode_pad: int = DECODE_PAD_TOKENS,
    q_chunk: int = 512,
    on_step=None,
) -> Tensor:
    """Returns uint8 [P, height, width, 3] on the CPU, one image per prompt."""
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    seeds = [seed] * len(prompts) if isinstance(seed, int) else list(seed)
    gh, gw = height // 16, width // 16
    ae_dtype = next(pipe.ae.parameters()).dtype

    texts = [""] * len(prompts) + prompts
    if geo_guidance != 0:
        texts += [f"{p}. {geo_prompt}" for p in prompts]
    if pipe.text_encoder is not None:
        pipe.encode_text(texts)
    ctx, ctx_ids = batched_prc_txt(torch.stack([pipe.ctx_of[t] for t in texts]).cuda())

    randn = torch.cat(
        [
            torch.randn(
                (1, 128, gh, gw),
                generator=torch.Generator(device="cuda").manual_seed(s),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for s in seeds
        ]
    )
    x, x_ids = batched_prc_img(randn)

    ref = ref_ids = keep = clean = None
    if ref_image:
        ref_images = ref_image.split(",") if isinstance(ref_image, str) else ref_image
        ref_images = ref_images if isinstance(ref_images, (list, tuple)) else [ref_images]
        tokens = []
        for k, image in enumerate(ref_images):
            pixels = default_images_prep(fit_reference(image, ref_max_pixels))[None].cuda()
            z = pipe.ae.encode(pixels.to(ae_dtype))[0]
            # FLUX.2 tells references apart by the t axis: 10, 20, ... (sampling.py:76)
            tokens.append(prc_img(z.to(torch.bfloat16), t_coord=torch.tensor([10 * (k + 1)])))
        ref = torch.cat([t for t, _ in tokens])[None]
        ref_ids = torch.cat([i for _, i in tokens])[None]
    if init_image is not None:
        init_pixels = load_pixels(init_image, width, height)
        clean = rearrange(pipe.ae.encode(init_pixels.to(ae_dtype)).to(torch.bfloat16), "b c h w -> b (h w) c")
        grid = keep_grid(width, height, keep_mask, keep_center)
        keep = grid.reshape(1, gh * gw, 1)

    geo = build_torus_geometry(pipe.model, x_ids, ctx_ids, (gh, gw), wrap, unanchor_text, ref_ids)
    timesteps = get_schedule(num_steps, x.shape[1])
    x = denoise_torus(
        pipe.model,
        x,
        ctx,
        geo,
        timesteps,
        guidance,
        geo_guidance,
        q_chunk,
        ref=ref,
        keep=keep,
        clean=clean,
        on_step=on_step,
    )

    x = rearrange(x, "b (h w) c -> b c h w", h=gh, w=gw)
    x = decode_torus(pipe.ae, x, wrap, decode_pad).float()
    if keep is not None and paste_kept_pixels:
        x = torch.where(grid.repeat_interleave(16, 0).repeat_interleave(16, 1), init_pixels, x)

    x = rearrange(x.clamp(-1, 1), "b c h w -> b h w c")
    return (127.5 * (x + 1.0)).cpu().byte()
