"""FLUX.2 klein image-to-image on the torus: a picture with a white hole in, a tile out.

    klein = Klein.load("flux.2-klein-9b", prompts=[FILL_PROMPT], system_prompt=SYSTEM_PROMPT)
    tile = klein(masked_picture, FILL_PROMPT, seed=0)          # PIL in, PIL out, same size

The recipe is the model's own -- for the distilled klein models 4 steps at guidance 1, which is one
forward pass per step -- and the picture goes in the way FLUX.2 takes any reference image: encoded,
and its tokens appended to the sequence. No mask channel, no init latent, no inpainting head, no
second prompt, no extra guidance branch. `system_prompt` is the text encoder's system turn, put in
front of every prompt it encodes. The one thing that is not stock is the attention: the
output's tokens sit on a torus (flux2/torus.py), so the model draws the left edge next to the right
edge and the top next to the bottom, and the decoder is handed the latent continued past its edges.
`rope` picks how: "nearest" moves the displacement to its nearest periodic copy, "quantized" rounds
the frequencies so the rotation itself repeats, plane by plane as `rules` says (both in torus.py).
`wrap=False` turns exactly that off and leaves stock FLUX.2, the baseline to compare against.

Nothing here assumes a GPU beyond `load`: `toy` builds the same object from small random weights on
the CPU, which draws noise but runs every line a real run does.
"""

import gc
import zlib
from dataclasses import dataclass, field

import torch
from einops import rearrange
from PIL import Image
from torch import Tensor

from flux2.autoencoder import AutoEncoder, AutoEncoderParams
from flux2.model import Flux2
from flux2.sampling import batched_prc_img, batched_prc_txt, default_images_prep, get_schedule, prc_img
from flux2.torus import build_torus_geometry, decode_torus, denoise_torus


class Klein:
    def __init__(
        self,
        model: Flux2,
        ae: AutoEncoder,
        ctx: dict[str, Tensor],  # prompt -> [1, L, D] text embedding; "" too when guidance != 1
        num_steps: int = 4,
        guidance: float = 1.0,
        wrap: bool = True,
        rope: str = "nearest",
        rules: list | None = None,
        unanchor_text: bool = False,
        q_chunk: int = 512,
    ):
        assert guidance == 1 or "" in ctx, "guidance != 1 needs the empty prompt's embedding"
        self.model, self.ae, self.ctx = model, ae, ctx
        self.num_steps, self.guidance = num_steps, guidance
        self.wrap, self.rope, self.rules = (wrap, wrap), rope, rules
        self.unanchor_text, self.q_chunk = unanchor_text, q_chunk

    @classmethod
    def load(
        cls,
        model_name: str,
        prompts: list[str],
        system_prompt: str | None = None,
        guidance: float = 1.0,
        **settings,
    ) -> "Klein":
        """The text encoder goes first and leaves before the flow model arrives: on an A40 the two
        together are 34 of 48 GB, and a run only ever needs a handful of prompts encoded once.
        `system_prompt` goes in front of all of them, the empty prompt of CFG included; None or ""
        is the bare user turn the klein models were trained on."""
        from flux2.util import load_ae, load_flow_model, load_text_encoder

        encoder = load_text_encoder(model_name, device=torch.device("cuda")).eval()
        texts = sorted(set(prompts) | ({""} if guidance != 1 else set()))
        with torch.no_grad():
            ctx = {
                text: encoder([text], system_message=system_prompt or None).to(torch.bfloat16)
                for text in texts
            }
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        model = load_flow_model(model_name, device=torch.device("cuda")).eval()
        return cls(model, load_ae(model_name).eval(), ctx, guidance=guidance, **settings)

    @classmethod
    def toy(
        cls, prompts: list[str], system_prompt: str | None = None, guidance: float = 1.0, **settings
    ) -> "Klein":
        """Random weights, CPU, seconds. The output is noise; what it proves is the plumbing."""

        @dataclass
        class ToyParams:
            in_channels: int = 128
            context_in_dim: int = 12
            hidden_size: int = 256  # 2 heads of 128: the head dim has to stay sum(axes_dim)
            num_heads: int = 2
            depth: int = 1
            depth_single_blocks: int = 1
            axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
            theta: int = 2000
            mlp_ratio: float = 3.0
            use_guidance_embed: bool = False

        torch.manual_seed(0)
        texts = sorted(set(prompts) | ({""} if guidance != 1 else set()))
        ctx = {
            text: torch.randn((1, 5, 12), generator=torch.Generator().manual_seed(zlib.crc32(chat.encode())))
            for text in texts
            for chat in [
                f"{system_prompt or ''}\n{text}"
            ]  # a stand-in for the encoder: the system turn counts
        }
        ae = AutoEncoder(AutoEncoderParams(ch=32)).eval()
        return cls(Flux2(ToyParams()).eval(), ae, ctx, guidance=guidance, **settings)

    @torch.no_grad()
    def __call__(self, reference: Image.Image, prompt: str, seed: int = 0) -> Image.Image:
        """One picture the size of `reference`, generated from pure noise with `reference` in view."""
        width, height = reference.size
        assert width % 16 == 0 and height % 16 == 0, f"{width}x{height} is not a multiple of 16"
        gh, gw = height // 16, width // 16
        weight = next(self.model.parameters())
        device, dtype = weight.device, weight.dtype
        ae_dtype = next(self.ae.parameters()).dtype

        ctx = self.ctx[prompt] if self.guidance == 1 else torch.cat((self.ctx[""], self.ctx[prompt]))
        ctx, ctx_ids = batched_prc_txt(ctx.to(device, dtype))

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((1, 128, gh, gw), generator=generator, dtype=dtype, device=device)
        x, x_ids = batched_prc_img(noise)

        # The reference at its own resolution: one token per 16 x 16 px, so the hole's edges land on
        # the same grid as the output's. FLUX.2 tells a reference from the output by the t axis: 10.
        pixels = default_images_prep(reference.convert("RGB"))[None].to(device, ae_dtype)
        ref, ref_ids = prc_img(self.ae.encode(pixels)[0].to(dtype), t_coord=torch.tensor([10]))
        ref, ref_ids = ref[None], ref_ids[None]

        geo = build_torus_geometry(
            self.model, x_ids, ctx_ids, (gh, gw), self.wrap, self.unanchor_text, ref_ids, self.rope, self.rules
        )
        timesteps = get_schedule(self.num_steps, x.shape[1])
        x = denoise_torus(self.model, x, ctx, geo, timesteps, self.guidance, self.q_chunk, ref=ref)

        x = rearrange(x, "b (h w) c -> b c h w", h=gh, w=gw)
        x = decode_torus(self.ae, x, self.wrap).float().clamp(-1, 1)
        return Image.fromarray((127.5 * (rearrange(x[0], "c h w -> h w c") + 1.0)).cpu().byte().numpy())
