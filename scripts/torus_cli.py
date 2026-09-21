"""Four-way seamless text-to-image with FLUX.2 klein base. The method is in src/flux2/torus.py.

    PYTHONPATH=src python scripts/torus_cli.py --prompts_file prompts.txt        (lines of `name: prompt`)
    PYTHONPATH=src python scripts/torus_cli.py --prompt "moss and small stones, top-down photo"

Baselines, same seed, same code path:
    --wrap_h=False --wrap_w=False     stock positions (tiles with visible seams)
    --geo_guidance=0                  plain CFG, two branches instead of three
    --unanchor_text=True              text no longer marks an origin on the torus (see torus.py)

Unlike scripts/cli.py this loads no content filter (that is a second, 24B model).
"""

from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

from flux2.sampling import batched_prc_img, batched_prc_txt, get_schedule
from flux2.torus import (
    DECODE_PAD_TOKENS,
    TORUS_PROMPT,
    build_torus_geometry,
    decode_torus,
    denoise_torus,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder


def read_prompts(path: str) -> list[tuple[str, str]]:
    """One prompt per line, `name: prompt`. Blank lines and lines starting with # are skipped."""
    pairs = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, prompt = line.split(":", 1)
            pairs.append((name.strip().replace(" ", "_"), prompt.strip()))
    return pairs


def main(
    prompts_file: str | None = None,
    prompt: str | None = None,
    batch_size: int = 4,  # prompts per forward; the network sees 3x this. Halve it on out-of-memory.
    seed: int = 0,  # every prompt starts from the same noise
    width: int = 256,
    height: int = 256,
    num_steps: int = 50,
    guidance: float = 4.0,
    geo_guidance: float = 2.0,  # untuned (the reference uses 6 for ERP on FLUX.2-dev); == guidance is plain CFG
    geo_prompt: str = TORUS_PROMPT,
    wrap_h: bool = True,
    wrap_w: bool = True,
    unanchor_text: bool = False,
    decode_pad: int = DECODE_PAD_TOKENS,
    q_chunk: int = 512,
    model_name: str = "flux.2-klein-base-4b",
    output_dir: str = "output",
):
    assert not FLUX2_MODEL_INFO[model_name]["guidance_distilled"], "real CFG needs an undistilled model"
    todo = read_prompts(prompts_file) if prompts_file else [("torus", prompt)]
    device = torch.device("cuda")
    wrap = (wrap_h, wrap_w)
    gh, gw = height // 16, width // 16
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    text_encoder = load_text_encoder(model_name, device=device).eval()
    model = load_flow_model(model_name, device=device).eval()
    ae = load_ae(model_name).eval()

    for start in range(0, len(todo), batch_size):
        names, prompts = zip(*todo[start : start + batch_size])
        print(f"[{start + 1}-{start + len(names)} of {len(todo)}] {', '.join(names)}")
        with torch.no_grad():
            texts = [""] * len(prompts) + list(prompts)
            if geo_guidance != 0:
                texts += [f"{p}. {geo_prompt}" for p in prompts]
            ctx, ctx_ids = batched_prc_txt(text_encoder(texts).to(torch.bfloat16))

            generator = torch.Generator(device="cuda").manual_seed(seed)
            randn = torch.randn((1, 128, gh, gw), generator=generator, dtype=torch.bfloat16, device="cuda")
            x, x_ids = batched_prc_img(randn.repeat(len(prompts), 1, 1, 1))

            geo = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), wrap, unanchor_text)
            timesteps = get_schedule(num_steps, x.shape[1])
            x = denoise_torus(model, x, ctx, geo, timesteps, guidance, geo_guidance, q_chunk)

            x = rearrange(x, "b (h w) c -> b c h w", h=gh, w=gw)
            x = decode_torus(ae, x, wrap, decode_pad).float()

        x = rearrange(x.clamp(-1, 1), "b c h w -> b h w c")
        x = (127.5 * (x + 1.0)).cpu().byte()
        for name, img in zip(names, x):
            stem = f"{name}_seed{seed}"
            path = out / f"{stem}_{len(list(out.glob(f'{stem}_*[0-9].png')))}.png"
            Image.fromarray(img.numpy()).save(path)
            # 2x2 tiling: all four seams meet in the middle, where they are easy to look at.
            Image.fromarray(img.repeat(2, 2, 1).numpy()).save(path.with_name(f"{path.stem}_tiled.png"))
            print(f"Saved {path} (+ _tiled)")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
