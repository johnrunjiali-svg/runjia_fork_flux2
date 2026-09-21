"""Four-way seamless images with FLUX.2 klein base. Method: src/flux2/torus.py. One-call API: src/flux2/torus_generate.py.

One prompt:
    PYTHONPATH=src python scripts/torus_cli.py --prompt "seamless rose pattern ..." --name roses --seeds 0,1
Many prompts (a file has one `name: prompt` per line; several files: a.txt,b.txt, each one a "set"):
    PYTHONPATH=src python scripts/torus_cli.py --prompts_file prompts.txt --seeds 0,1,2,3 --run_name first
    bash scripts/torus_multi_gpu.sh                      the same, sharded over 8 GPUs, then the gallery

Everything else is an argument of `generate` and is passed through, e.g.
    --wrap=False,False                stock positions (tiles with visible seams): the baseline
    --geo_guidance=0                  plain CFG, two branches instead of three
    --unanchor_text=True              text no longer marks an origin on the torus (see torus.py)
    --ref_image a.png                 FLUX.2 image-to-image: a.png is a reference the prompt can talk about
    --init_image a.png --keep_center 0.7      the middle 70% x 70% of a.png stays untouched, the band
                                              around it is generated so that the picture tiles
    --init_image a.png --keep_mask m.png      the same with any region: white = untouched
    --width 512 --height 512

Unlike scripts/cli.py this loads no content filter (that is a second, 24B model).
"""

import inspect
import json
from pathlib import Path

from PIL import Image

from flux2.torus import TORUS_PROMPT
from flux2.torus_generate import TorusPipe, fit_reference, generate, keep_grid, save_png


def read_prompts(path: str) -> list[tuple[str, str]]:
    """One prompt per line, `name: prompt`. Blank lines and lines starting with # are skipped."""
    pairs = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, prompt = line.split(":", 1)
            pairs.append((name.strip().replace(" ", "_"), prompt.strip()))
    return pairs


def main(
    prompt: str | None = None,  # one prompt, saved under <run>/single/<name>/ ...
    name: str = "image",
    prompts_file: str = "prompts.txt",  # ... or files of prompts, used when --prompt is not given
    seeds: int | tuple = 0,  # one seed or several (0,1,2,3); every prompt is run with every seed
    run_name: str = "run",
    shard: int = 0,  # this process takes jobs[shard::num_shards]; scripts/torus_multi_gpu.sh sets these
    num_shards: int = 1,
    batch_size: int = 6,  # (prompt, seed) jobs per forward; the network sees 3x this. Halve it on out-of-memory.
    model_name: str = "flux.2-klein-base-9b",
    output_dir: str = "output",
    **settings,  # arguments of flux2.torus_generate.generate
):
    """Writes  <output_dir>/<run_name>/<set>/<name>/seed<k>.png  and  seed<k>_tiled.png,
    plus config.json and one manifest_<shard>.jsonl that scripts/torus_gallery.py turns into gallery.md."""
    unknown = set(settings) - set(inspect.signature(generate).parameters)
    assert not unknown, f"not arguments of generate: {unknown}"  # before minutes of model loading, not after
    seeds = [seeds] if isinstance(seeds, int) else list(seeds)
    if prompt is not None:
        named = [("single", name, prompt)]
    else:
        named = [(Path(f).stem, n, p) for f in prompts_file.split(",") for n, p in read_prompts(f)]
    jobs = [(set_name, n, p, s) for set_name, n, p in named for s in seeds][shard::num_shards]

    run = Path(output_dir) / run_name
    run.mkdir(parents=True, exist_ok=True)
    config = {"model_name": model_name, "geo_prompt": TORUS_PROMPT, **settings}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    manifest = (run / f"manifest_{shard}.jsonl").open("w")

    if shard == 0:  # what went in; the gallery shows these on top
        size = settings.get("width", 256), settings.get("height", 256)
        if "ref_image" in settings or "init_image" in settings:
            (run / "inputs").mkdir(exist_ok=True)
        for k, ref in enumerate(settings.get("ref_image", "").split(",") if "ref_image" in settings else []):
            ref = fit_reference(
                ref, settings.get("ref_max_pixels", 512**2)
            )  # exactly what the network is shown
            ref.save(run / "inputs" / f"ref{k + 1}.png")
        if "init_image" in settings:
            Image.open(settings["init_image"]).convert("RGB").resize(size).save(run / "inputs" / "init.png")

    pipe = TorusPipe(model_name)
    geo_prompt = settings.get("geo_prompt", TORUS_PROMPT)
    pipe.encode_text([t for _, _, p, _ in jobs for t in ("", p, f"{p}. {geo_prompt}")])
    pipe.drop_text_encoder()

    if shard == 0 and "init_image" in settings:
        grid = keep_grid(*size, settings.get("keep_mask"), settings.get("keep_center", 0.0))
        print(f"keeping {int(grid.sum())} of {grid.numel()} tokens untouched")
        white = grid.repeat_interleave(16, 0).repeat_interleave(16, 1).cpu().numpy()
        Image.fromarray((white * 255).astype("uint8")).save(run / "inputs" / "keep.png")

    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        print(f"shard {shard}: jobs {start + 1}-{start + len(batch)} of {len(jobs)}")
        images = generate(pipe, [p for _, _, p, _ in batch], [s for _, _, _, s in batch], **settings)
        for (set_name, n, p, s), img in zip(batch, images):
            path = run / set_name / n / f"seed{s}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_png(img, path, tiled=True)  # 2x2 tiling: the four seams meet in the middle, easy to look at
            manifest.write(json.dumps({"set": set_name, "name": n, "seed": s, "prompt": p}) + "\n")
            manifest.flush()

    if (
        num_shards == 1
    ):  # a lone process finishes its own run; the multi-GPU launcher does this after all shards
        from torus_gallery import main as write_gallery

        write_gallery(str(run))


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
