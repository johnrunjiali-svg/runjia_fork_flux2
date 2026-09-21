"""Four-way seamless text-to-image with FLUX.2 klein base. The method is in src/flux2/torus.py.

    PYTHONPATH=src python scripts/torus_cli.py --prompts_file prompts.txt --seeds 0,1,2,3 --run_name first
    bash scripts/torus_multi_gpu.sh                      the same, sharded over 8 GPUs, then the gallery

A prompts file has one `name: prompt` per line.

Baselines, same seed, same code path:
    --wrap_h=False --wrap_w=False     stock positions (tiles with visible seams)
    --geo_guidance=0                  plain CFG, two branches instead of three
    --unanchor_text=True              text no longer marks an origin on the torus (see torus.py)

Unlike scripts/cli.py this loads no content filter (that is a second, 24B model).
"""

import json
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

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
    prompts_file: str = "prompts.txt",  # or several: a.txt,b.txt  Each file is a "set", named by its stem.
    seeds: int | tuple = 0,  # one seed or several (0,1,2,3); every prompt is run with every seed
    run_name: str = "run",
    shard: int = 0,  # this process takes jobs[shard::num_shards]; scripts/torus_multi_gpu.sh sets these
    num_shards: int = 1,
    batch_size: int = 6,  # (prompt, seed) jobs per forward; the network sees 3x this. Halve it on out-of-memory.
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
    model_name: str = "flux.2-klein-base-9b",
    output_dir: str = "output",
):
    """Writes  <output_dir>/<run_name>/<set>/<name>/seed<k>.png  and  seed<k>_tiled.png,
    plus config.json and one manifest_<shard>.jsonl that scripts/torus_gallery.py turns into gallery.md."""
    assert not FLUX2_MODEL_INFO[model_name]["guidance_distilled"], "real CFG needs an undistilled model"
    config = dict(locals())
    files = prompts_file.split(",")
    seeds = [seeds] if isinstance(seeds, int) else list(seeds)
    jobs = [
        (Path(f).stem, name, prompt, seed)
        for f in files
        for name, prompt in read_prompts(f)
        for seed in seeds
    ][shard::num_shards]

    device = torch.device("cuda")
    wrap = (wrap_h, wrap_w)
    gh, gw = height // 16, width // 16
    run = Path(output_dir) / run_name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(config, indent=2))
    manifest = (run / f"manifest_{shard}.jsonl").open("w")

    # All the text first, then the text encoder leaves the GPU: on cards without FP8 (A40) the 8B
    # encoder is dequantized to 16 GB of bf16, and the 9B flow model wants 18 GB of its own.
    text_encoder = load_text_encoder(model_name, device=device).eval()
    texts = sorted({t for _, _, p, _ in jobs for t in ("", p, f"{p}. {geo_prompt}")})
    ctx_of = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 8), desc=f"shard {shard} text"):
            for t, c in zip(texts[i : i + 8], text_encoder(texts[i : i + 8]).to(torch.bfloat16).cpu()):
                ctx_of[t] = c
    del text_encoder
    torch.cuda.empty_cache()

    model = load_flow_model(model_name, device=device).eval()
    ae = load_ae(model_name).eval()

    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        print(f"shard {shard}: jobs {start + 1}-{start + len(batch)} of {len(jobs)}")
        with torch.no_grad():
            texts = [""] * len(batch) + [p for _, _, p, _ in batch]
            if geo_guidance != 0:
                texts += [f"{p}. {geo_prompt}" for _, _, p, _ in batch]
            ctx, ctx_ids = batched_prc_txt(torch.stack([ctx_of[t] for t in texts]).to(device))

            # One generator per job, so an image depends on its seed and not on who shares its batch.
            randn = torch.cat(
                [
                    torch.randn(
                        (1, 128, gh, gw),
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    for _, _, _, seed in batch
                ]
            )
            x, x_ids = batched_prc_img(randn)

            geo = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), wrap, unanchor_text)
            timesteps = get_schedule(num_steps, x.shape[1])
            x = denoise_torus(model, x, ctx, geo, timesteps, guidance, geo_guidance, q_chunk)

            x = rearrange(x, "b (h w) c -> b c h w", h=gh, w=gw)
            x = decode_torus(ae, x, wrap, decode_pad).float()

        x = rearrange(x.clamp(-1, 1), "b c h w -> b h w c")
        x = (127.5 * (x + 1.0)).cpu().byte()
        for (set_name, name, prompt, seed), img in zip(batch, x):
            path = run / set_name / name / f"seed{seed}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img.numpy()).save(path)
            # 2x2 tiling: all four seams meet in the middle, where they are easy to look at.
            Image.fromarray(img.repeat(2, 2, 1).numpy()).save(path.with_name(f"seed{seed}_tiled.png"))
            record = {"set": set_name, "name": name, "seed": seed, "prompt": prompt}  # gallery sorts by these
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
