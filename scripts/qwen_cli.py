"""Batch runs of qwen_torus.generate from the command line, one picture per prompt.

    PYTHONPATH=src uv run python scripts/qwen_cli.py --prompt "seamless rose pattern" --out output/qwen
    PYTHONPATH=src uv run python scripts/qwen_cli.py --prompts prompts/geo.txt --seed 0 --width 1024 --height 1024
    ... --pe '{"mode": "periodic", "min_cycles": 0.5, "low_freq": "keep"}'       (any PeConfig field)
    ... --cond_images masked.png --prompt "Fill the white areas ..."              (image-to-image)
    ... --init_image tile.png --keep_center 0.6 --t_start 0.6                     (keep the middle, redraw the band)

Every image is written as <out>/<slug>_<seed>.png and <slug>_<seed>_tiled.png (2x2 copies), with the
settings next to it as <slug>_<seed>.json.
"""

import json
import re
import time
from pathlib import Path

from PIL import Image

from qwen_torus import PeConfig, QwenTorusPipe, generate
from qwen_torus.generate import measure_decoder_receptive_field


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48] or "image"


def main(
    prompt: str | None = None,
    prompts: str | None = None,  # a text file, one prompt per line
    out: str = "output/qwen",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 1.0,
    geo_guidance: float = 2.0,
    pe: str | dict | None = None,  # json or a dict of PeConfig fields
    cond_images: str | None = None,  # "a.png,b.png"
    init_image: str | None = None,
    keep_mask: str | None = None,
    keep_center: float = 0.0,
    t_start: float | None = None,
    model_id: str = "Qwen/Qwen-Image-2.1",
    cpu_offload: bool = True,
    vae_tiling: bool = False,
    measure_vae: bool = False,  # print the decoder's receptive field and stop
    **settings,
):
    pipe = QwenTorusPipe(model_id, cpu_offload=cpu_offload, vae_tiling=vae_tiling)
    if measure_vae:
        print("decoder receptive field (latent px):", measure_decoder_receptive_field(pipe.vae))
        return
    texts = [prompt] if prompt else []
    if prompts:
        texts += [line.strip() for line in Path(prompts).read_text().splitlines() if line.strip()]
    assert texts, "give --prompt or --prompts"
    cfg = PeConfig.from_dict(json.loads(pe) if isinstance(pe, str) else pe)
    refs = cond_images.split(",") if cond_images else None
    Path(out).mkdir(parents=True, exist_ok=True)
    for text in texts:
        started = time.time()
        tile = generate(
            pipe, text, seed=seed, width=width, height=height, num_steps=num_steps, guidance=guidance,
            geo_guidance=geo_guidance, pe=cfg, cond_images=refs, init_image=init_image, keep_mask=keep_mask,
            keep_center=keep_center, t_start=t_start, **settings,
        )  # fmt: skip
        name = Path(out) / f"{slug(text)}_{seed}"
        image = Image.fromarray(tile.numpy())
        image.save(f"{name}.png")
        Image.fromarray(tile.repeat(2, 2, 1).numpy()).save(f"{name}_tiled.png")
        record = dict(prompt=text, seed=seed, width=width, height=height, num_steps=num_steps, guidance=guidance,
                      geo_guidance=geo_guidance, pe=cfg.to_dict(), cond_images=refs, init_image=init_image,
                      keep_mask=keep_mask, keep_center=keep_center, t_start=t_start, model=model_id,
                      seconds=round(time.time() - started, 1), **settings)  # fmt: skip
        Path(f"{name}.json").write_text(json.dumps(record, indent=2))
        print(f"{name}.png  ({record['seconds']} s)")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
