"""One call from prompt (+ optional pictures) to a four-way seamless image with Qwen-Image 2.1.

    from qwen_torus import QwenTorusPipe, generate, PeConfig
    pipe = QwenTorusPipe()                                           # Qwen/Qwen-Image-2.1, bf16, cpu offload
    tile = generate(pipe, "seamless rose pattern", seed=0, cond_images=["masked.png"])   # uint8 [H, W, 3]

The sampler is the stock pipeline's loop (diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
reduced to what is needed and opened up where it has to be:

- Three guidance branches, as in src/flux2/torus.py: v = v_uncond + guidance (v_cond - v_uncond)
  + geo_guidance (v_geo - v_cond), where v_geo is the prompt with TORUS_PROMPT appended. Qwen-Image
  2.1 is not guidance-distilled, so every branch is a full network evaluation; `guidance_weights`
  turns the two numbers into one weight per branch and a branch with weight zero is not run. The
  model card samples without guidance at all (guidance = 1), which drops the unconditional branch.
- Each branch is its own forward pass at batch size 1 with its own KV cache: prompts differ in
  length, and the model's cache of the prefix (text + condition images, keys stored after RoPE)
  is exactly what makes the later steps cheap -- only the target image's rows are recomputed.
- Position enters through `qwen_torus.attention.install`: the rope wrapper and, when the config
  needs it, the hand-written attention. Nothing in the weights changes.
- `keep` / `t_start` / `clean` are the inpainting-as-constraint of torus.py: kept tokens are
  overwritten after every step with the clean latent noised to the current time.
- Decoding pads the latent circularly by the decoder's receptive field so the pixel seam is as
  continuous as the latent one. The VAE returns RGBA; it is composited over white.
"""

import hashlib
import math
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from .attention import TorusContext, install
from .prep import TORUS_PROMPT, fit_area, seam_ratio
from .rope import PeConfig, build_geometry

# One-sided receptive field of the VAE decoder in latent pixels (= tokens: 2.1 is unpatched, 16 px
# each): conv_in, the mid block's four convs and the first up block's six at latent resolution, then
# the later stages at 2x..16x contribute half as much each -- about 17.6. The mid block also holds one
# global attention, so this is the local part only, as it was for FLUX (`measure_decoder_receptive_field`).
DECODE_PAD_TOKENS = 18

# Where `generate` starts the sampler when it is given an init image and no explicit t_start.
INIT_T_START = 0.6

# No wrapping, stock table, stock fused processor: what `generate` leaves installed, and what the web
# page's "seamless" box sends when it is off (together with seamless strength 0).
STOCK = PeConfig(mode="none", wrap_h=False, wrap_w=False)

# Qwen/Qwen-Image-2.1/scheduler/scheduler_config.json; read from the loaded scheduler when there is one.
SCHEDULER_DEFAULTS = dict(
    base_image_seq_len=256, max_image_seq_len=8192, base_shift=0.5, max_shift=0.9, shift_terminal=0.02
)


class QwenTorusPipe:
    """The loaded pipeline plus what this package adds: the shared attention context and a cache of
    prompt embeddings (the Qwen3-VL encoder is 8B parameters; a prompt is encoded once per set of
    condition images)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-2.1",
        device: str = "cuda",
        cpu_offload: bool = True,  # text encoder (16 GB) -> transformer (14 GB) -> VAE, one on the GPU at a time
        vae_tiling: bool = False,  # for 2048^2: the decoder on a (128 + 36)^2 padded latent is large
        dtype=torch.bfloat16,
    ):
        from diffusers import QwenImage21Pipeline

        self.model_name = model_id
        self.pipe = QwenImage21Pipeline.from_pretrained(model_id, dtype=dtype)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload(device=device)
        else:
            self.pipe.to(device)
        if vae_tiling:
            self.pipe.vae.enable_tiling()
        self.device = torch.device(device)
        self.dtype = dtype
        self.ctx = TorusContext()
        self.embeds_of: dict[tuple, tuple[Tensor, Tensor]] = {}

    @property
    def transformer(self):
        return self.pipe.transformer

    @property
    def vae(self):
        return self.pipe.vae

    @property
    def image_processor(self):
        return self.pipe.image_processor

    @property
    def scheduler_config(self) -> dict:
        return dict(self.pipe.scheduler.config)

    @torch.no_grad()
    def encode_prompt(self, text: str, images: list[Image.Image] | None) -> tuple[Tensor, Tensor]:
        """[1, L, 4096] embeddings and the [1, L] bool mask of the vision slots, on the device. The
        encoder reads the condition images, so they are part of the key."""
        key = (text, *(hashlib.sha1(im.tobytes()).hexdigest() + str(im.size) for im in images or []))
        if key not in self.embeds_of:
            embeds, _, pad_mask = self.pipe.encode_prompt(
                prompt=text, image=images or None, device=self.device
            )
            if len(self.embeds_of) >= 64:
                self.embeds_of.pop(next(iter(self.embeds_of)))
            self.embeds_of[key] = (embeds.cpu(), pad_mask.cpu())
        embeds, pad_mask = self.embeds_of[key]
        return embeds.to(self.device), pad_mask.to(self.device)

    def free(self):
        """Offload every model (what the stock pipeline does at the end of a call)."""
        self.pipe.maybe_free_model_hooks()


def guidance_weights(guidance: float, geo_guidance: float) -> list[float]:
    """Weights of [v_uncond, v_cond, v_geo]: (1 - guidance, guidance - geo_guidance, geo_guidance).
    Zero means the branch is not evaluated (torus.py has the full argument)."""
    return [1.0 - guidance, guidance - geo_guidance, geo_guidance]


def schedule(num_steps: int, seq_len: int, t_start: float = 1.0, config: dict | None = None) -> list[float]:
    """num_steps + 1 noise levels from t_start down to 0, the stock Qwen-Image 2.1 schedule when
    t_start = 1: sigmas evenly spaced in [1, 1/n], bent by the resolution-dependent exponential shift
    s(u) = e^mu / (e^mu + 1/u - 1) with mu linear in the token count, then stretched so the last one
    is shift_terminal (0.02). The shift inverts in closed form, so for t_start < 1 the first sigma is
    placed at exactly t_start and the rest keep the same bending."""
    assert 0.0 < t_start <= 1.0, t_start
    c = {**SCHEDULER_DEFAULTS, **{k: v for k, v in (config or {}).items() if k in SCHEDULER_DEFAULTS}}
    m = (c["max_shift"] - c["base_shift"]) / (c["max_image_seq_len"] - c["base_image_seq_len"])
    mu = c["base_shift"] + m * (seq_len - c["base_image_seq_len"])
    shift = lambda u: math.exp(mu) / (math.exp(mu) + (1.0 / u - 1.0))  # noqa: E731
    u_last = 1.0 / num_steps
    scale = (1.0 - shift(u_last)) / (1.0 - c["shift_terminal"]) if c["shift_terminal"] else 1.0
    if t_start == 1.0:
        u_first = 1.0
    else:
        s_first = 1.0 - (1.0 - t_start) * scale  # what must come out of the shift to stretch to t_start
        u_first = 1.0 / (1.0 + math.exp(mu) * (1.0 / s_first - 1.0))
    u = torch.linspace(u_first, u_last, num_steps, dtype=torch.float64).tolist()
    return [1.0 - (1.0 - shift(x)) / scale for x in u] + [0.0]


def pixels_of(pipe, image: str | Image.Image, width: int, height: int) -> Tensor:
    """[1, 4, 1, height, width] RGBA in [-1, 1] on the device, the VAE's input. Resized, not cropped."""
    image = Image.open(image) if isinstance(image, (str, Path)) else image
    image = image.convert("RGBA")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return (
        pipe.image_processor.preprocess(image, width=width, height=height)
        .unsqueeze(2)
        .to(pipe.device, pipe.dtype)
    )


def _latent_stats(pipe, like: Tensor) -> tuple[Tensor, Tensor]:
    c = pipe.vae.config
    shape = (1, -1, 1, 1, 1)
    return (
        torch.tensor(c.latents_mean, device=like.device, dtype=like.dtype).view(shape),
        torch.tensor(c.latents_std, device=like.device, dtype=like.dtype).view(shape),
    )


@torch.no_grad()
def encode_image(pipe, pixels: Tensor) -> Tensor:
    """[1, 4, 1, H, W] pixels -> [1, (H/16)(W/16), C] packed, normalised latents (the mode, not a sample)."""
    z = pipe.vae.encode(pixels).latent_dist.mode()
    mean, std = _latent_stats(pipe, z)
    z = (z - mean) / std
    return z.flatten(2).transpose(1, 2)


@torch.no_grad()
def decode_latents(
    pipe, x: Tensor, grid: tuple[int, int], wrap=(True, True), pad: int = DECODE_PAD_TOKENS
) -> Tensor:
    """[1, N, C] packed latents on a gh x gw grid -> [1, 4, 16 gh, 16 gw] RGBA in [-1, 1]. The decoder's
    convolutions zero-pad, which would put the seam back at the pixel level: hand it the torus unrolled
    a little past each wrapped edge and crop what comes back."""
    gh, gw = grid
    z = x.transpose(1, 2).reshape(1, -1, gh, gw).to(pipe.vae.dtype)
    ph, pw = min(pad, gh - 1) * wrap[0], min(pad, gw - 1) * wrap[1]
    z = F.pad(z, (pw, pw, ph, ph), mode="circular").unsqueeze(2)
    mean, std = _latent_stats(pipe, z)
    out = pipe.vae.decode(z * std + mean).sample[:, :, 0].float()
    return out[..., 16 * ph : out.shape[-2] - 16 * ph, 16 * pw : out.shape[-1] - 16 * pw]


def to_rgb(rgba: Tensor) -> Tensor:
    """[B, 4, H, W] in [-1, 1] -> uint8 [B, H, W, 3] on the CPU, alpha composited over white (what the
    model's own vision encoder is shown for an RGBA input)."""
    rgb, alpha = rgba[:, :3].clamp(-1, 1), ((rgba[:, 3:] + 1) / 2).clamp(0, 1)
    rgb = rgb * alpha + (1 - alpha)
    return (127.5 * (rgb + 1)).round().clamp(0, 255).permute(0, 2, 3, 1).cpu().byte()


def keep_grid(pipe, width: int, height: int, keep_mask=None, keep_center: float = 0.0) -> Tensor:
    """[gh, gw] bool: tokens that must stay untouched. From a mask (white = kept; a token is kept only
    if all its 16x16 pixels are white), else the centred rectangle covering `keep_center` of each side."""
    gh, gw = height // 16, width // 16
    if keep_mask is not None:
        white = pixels_of(pipe, keep_mask, width, height)[:, :3, 0].float().mean(1, keepdim=True) > 0.99
        return F.avg_pool2d(white.float(), 16)[0, 0] == 1
    mh, mw = round(gh * (1 - keep_center) / 2), round(gw * (1 - keep_center) / 2)
    grid = torch.zeros(gh, gw, dtype=torch.bool, device=pipe.device)
    grid[mh : gh - mh, mw : gw - mw] = True
    return grid


def _joint_mask(pad_mask: Tensor) -> Tensor:
    """The vision-slot mask of the VLM sequence expanded to the joint sequence: each slot is 2x2 tokens."""
    return torch.repeat_interleave(pad_mask[0], torch.where(pad_mask[0], 4, 1))


@torch.no_grad()
def generate(
    pipe: QwenTorusPipe,
    prompt: str,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 1.0,  # true CFG scale; 1 is what the model card samples at, and skips the uncond branch
    geo_guidance: float = 2.0,  # weight of the torus sentence; 0 skips it; == guidance is plain CFG on the long prompt
    geo_prompt: str = TORUS_PROMPT,
    negative_prompt: str = "",  # the unconditional branch's prompt ("" is encoded as " ", Qwen has no bos)
    pe: PeConfig | dict | None = None,
    cond_images: list
    | None = None,  # condition images: PIL or paths. The first is resized to the output size
    ref_max_pixels: int = 1024**2,  # the others are shrunk to this area (each is 1 token per 16x16 px)
    init_image=None,  # the picture whose kept region stays untouched
    keep_mask=None,
    keep_center: float = 0.0,
    t_start: float | None = None,  # noise level to start at; None = INIT_T_START with an init image, else 1
    paste_kept_pixels: bool = True,
    decode_pad: int = DECODE_PAD_TOKENS,
    on_step=None,  # on_step(steps_done, steps_total)
    info: dict | None = None,  # filled with what the run measured: seam_latent, seam_pixels (rows, columns)
) -> Tensor:
    """Returns uint8 [height, width, 3] on the CPU. Sizes must be multiples of 32: a token is 16 px and
    the vision encoder's slots are 2x2 tokens.

    `info["seam_latent"]` is `seam_ratio` of the final latent grid -- the transformer's own seam, before
    the decoder can hide or add anything -- and `info["seam_pixels"]` the same on the picture."""
    assert width % 32 == 0 and height % 32 == 0, f"{width}x{height}: sizes must be multiples of 32"
    pe = PeConfig.load(pe)  # a PeConfig, a dict, a json string or the path of a configs/pe/*.json file
    gh, gw = height // 16, width // 16
    n_t = gh * gw
    device, dtype = pipe.device, pipe.dtype
    transformer = pipe.transformer

    # 1. Branches: only those with a nonzero weight are encoded and run, in the order (uncond, cond, geo).
    texts = [negative_prompt, prompt, f"{prompt} {geo_prompt}".strip()]
    branches = [(w, t) for w, t in zip(guidance_weights(guidance, geo_guidance), texts) if w != 0]
    assert branches, "guidance and geo_guidance leave nothing to evaluate"

    # 2. Condition images: the vision encoder sees them (RGBA flattened over white by the pipeline) and
    #    the VAE encodes them into prefix tokens. The first one lines up token for token with the output.
    refs = []
    for k, image in enumerate(cond_images or []):
        image = Image.open(image) if isinstance(image, (str, Path)) else image
        image = image.convert("RGBA")
        refs.append(
            image.resize((width, height), Image.Resampling.LANCZOS)
            if k == 0
            else fit_area(image, ref_max_pixels)
        )
    encoded = [pipe.encode_prompt(t, refs) for _, t in branches]
    cond_latents = [encode_image(pipe, pixels_of(pipe, r, *r.size)) for r in refs]
    cond_shapes = [(1, r.height // 16, r.width // 16) for r in refs]
    prefix_latents = torch.cat(cond_latents, dim=1) if cond_latents else None

    # 3. Noise, and the picture to keep, if any.
    generator = torch.Generator(device=device).manual_seed(int(seed))
    c = transformer.config.in_channels
    noise = torch.randn((1, c, 1, gh, gw), generator=generator, device=device, dtype=dtype)
    noise = noise.flatten(2).transpose(1, 2).float()  # [1, N_t, C]
    keep = clean = init_pixels = None
    if init_image is not None:
        init_pixels = pixels_of(pipe, init_image, width, height)
        clean = encode_image(pipe, init_pixels).float()
        grid = keep_grid(pipe, width, height, keep_mask, keep_center)
        keep = grid.reshape(1, n_t, 1)
    if t_start is None:
        t_start = INIT_T_START if init_image is not None else 1.0
    t_start = float(t_start)
    assert init_image is not None or t_start == 1.0, "t_start < 1 starts from the init image; give one"
    x = noise if t_start == 1.0 else (1 - t_start) * clean + t_start * noise

    # 4. Position: the rope wrapper and, if needed, the hand-written attention. Whatever happens, the
    #    transformer is handed back in its stock state, so `pipe.pipe(...)` after this is plain Qwen.
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    try:
        rope = install(transformer, pipe.ctx, pe)
        pipe.ctx.geo = build_geometry(rope, pe, (gh, gw), device)
        timesteps = schedule(num_steps, n_t, t_start, pipe.scheduler_config)

        # 5. Euler flow matching, one forward per branch per step, prefix KV cached after the first step.
        caches = [QwenImage21KVCache(len(transformer.transformer_blocks)) for _ in branches]
        img_shapes = [[*cond_shapes, (1, gh, gw)]]
        for step, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            v = torch.zeros_like(x)
            for (weight, _), (embeds, pad_mask), cache in zip(branches, encoded, caches):
                pipe.ctx.prefix_is_text = ~_joint_mask(pad_mask)
                hidden = (
                    x.to(dtype)
                    if prefix_latents is None
                    else torch.cat([prefix_latents.to(dtype), x.to(dtype)], dim=1)
                )
                pred = transformer(
                    hidden_states=hidden,
                    encoder_hidden_states=embeds,
                    timestep=torch.full((1,), t_curr, device=device, dtype=torch.float32),
                    img_shapes=img_shapes,
                    img_mask=torch.cat([pad_mask, pad_mask.new_ones(1, n_t // 4)], dim=1),
                    kv_cache=cache,
                    kv_cache_mode="extract" if step == 0 else "cached",
                    return_dict=False,
                )[0][:, -n_t:].float()
                v = v + weight * pred
            x = x + (t_prev - t_curr) * v
            if keep is not None:
                x = torch.where(keep, (1 - t_prev) * clean + t_prev * noise, x)
            if on_step is not None:
                on_step(step + 1, num_steps)

        # 6. Pixels.
        out = decode_latents(pipe, x, (gh, gw), pe.wrap, decode_pad)
    finally:
        install(transformer, pipe.ctx, STOCK)
        pipe.ctx.geo = None
    if keep is not None and paste_kept_pixels:
        mask = grid.repeat_interleave(16, 0).repeat_interleave(16, 1)
        out = torch.where(mask, init_pixels[:, :, 0].float(), out)
    pipe.free()
    rgb = to_rgb(out)[0]
    if info is not None:
        info["seam_latent"] = seam_ratio(x[0].reshape(gh, gw, -1))
        info["seam_pixels"] = seam_ratio(rgb)
    return rgb


@torch.no_grad()
def measure_decoder_receptive_field(
    vae, grid: int = 64, channel: int = 0, eps: float = 1e-3, device=None
) -> int:
    """One-sided receptive field of the decoder in latent pixels, measured: decode a zero latent and the
    same with one latent pixel perturbed, and see how far the output differs. Run once on the real VAE
    (seconds on a GPU) to check DECODE_PAD_TOKENS; the mid block's global attention makes the strict
    answer "everything", so `eps` cuts off the tail.

    `device` is where the VAE *runs*. Under cpu offload its weights sit on the CPU until `decode` is
    called and accelerate's hook moves the module -- not the inputs -- so the input has to be built on
    the execution device, which the hook knows."""
    hook = getattr(vae, "_hf_hook", None)
    device = device or getattr(hook, "execution_device", None) or next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    z = torch.zeros(1, vae.config.z_dim, 1, grid, grid, device=device, dtype=dtype)
    base = vae.decode(z).sample.float()
    z[0, channel, 0, grid // 2, grid // 2] = 1.0
    diff = (vae.decode(z).sample.float() - base).abs().amax(dim=(0, 1, 2))  # [H, W] pixels
    rows = torch.where(diff.amax(1) > eps * diff.max())[0]
    reach = max(grid // 2 * 16 - rows.min().item(), rows.max().item() - grid // 2 * 16)
    return math.ceil(reach / 16)
