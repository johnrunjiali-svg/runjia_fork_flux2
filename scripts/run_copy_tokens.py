"""Experiment 4, on the server: the copy-token attention swept over its band, one process per GPU.

Inpainting, the task of experiment 3 (centre crop on a white canvas, rolled so the hole is a cross):

    uv run python scripts/run_copy_tokens.py                                  all tiles, bands 0, 4, 16 and the RoPE control
    uv run python scripts/run_copy_tokens.py --bands 0,1,2,4,8,16,32          the sweep; each band is one more generation per job
    uv run python scripts/run_copy_tokens.py --compare_rope False             drop the nearest-copy RoPE column
    uv run python scripts/run_copy_tokens.py --task 1_seam_fix                the tile itself with the cross, as in experiment 1
    uv run python scripts/run_copy_tokens.py --hole 64                        px of white across each seam (the frame is hole / 2)
    uv run python scripts/run_copy_tokens.py --only 00,03,19 --gpus 0,1       as in run_experiments.py; also --tiles, --prompt,
                                                                              --system_prompt, --guidance, --seed, --run_name

Text to image, the prompt alone:

    uv run python scripts/run_copy_tokens.py --task t2i --prompts prompts.txt       one prompt per line (blank and # lines skipped)
    uv run python scripts/run_copy_tokens.py --task t2i --prompts "moss on stone"   or one prompt
    uv run python scripts/run_copy_tokens.py --task t2i --prompts p.txt --size 1024 --system_prompt none

The system turn is SYSTEM_PROMPT (inpaint) or T2I_SYSTEM_PROMPT (t2i) unless --system_prompt text|file.txt|none.

    uv run python scripts/run_copy_tokens.py --dry_run                        no model: the inpainting inputs only (laptop)
    uv run python scripts/run_copy_tokens.py --toy --only 0,1 --bands 0,1     random toy weights on the CPU: the plumbing

What the experiment is is in src/seamless/copy_tokens.py; the method in src/flux2/copy_tokens.py.

    output/<run_name>/
        index.html, summary.csv         medians per band, and every job with its numbers per band
        config.json, logs/gpu<k>.log
        <job>/                          a tile's name, or <k>_<first words of the prompt>
            1_*.png ...                 every panel as its own PNG: the inputs (inpaint), then result and 2x2 per band
            sheet.png, thumb.jpg        all of them side by side at full resolution, seams marked
            run.json
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from run_experiments import REPO, as_list, fetch_weights, find_gpus, find_tiles
from seamless.copy_tokens import T2I_SYSTEM_PROMPT, slug, write_report
from seamless.experiments import EXPERIMENTS, FILL_PROMPT, SYSTEM_PROMPT


def read_text(value: str | None, default: str) -> str:
    """A setting that may be text, a .txt file, "none" for nothing, or unset for the default."""
    if value is None:
        return default
    if str(value).lower() == "none":
        return ""
    return Path(value).read_text().strip() if str(value).endswith(".txt") else str(value)


def main(
    task: str = "3_outpaint_cross",  # 1_seam_fix, 2_outpaint_frame, 3_outpaint_cross (inpainting) or t2i
    tiles: str = "principled/tiles_1024",  # inpainting: folders, files or globs, comma separated
    only=None,  # of those, the ones whose leading number or name matches: 00,03,19 or rose,ivy
    prompts: str | None = None,  # t2i: a .txt with one prompt per line, or one prompt
    size: int = 1024,  # t2i: the square picture
    bands="0,4,16",  # copy band in tokens (16 px each); 0 is stock attention
    compare_rope: bool = True,  # one more column: the nearest-copy RoPE of experiments 1-3
    hole: int = 128,  # inpainting: px of white across each seam; the cross of tasks 1 and 3, twice the frame of 2
    prompt: str = FILL_PROMPT,  # inpainting: the instruction, or a .txt file that holds it
    system_prompt: str | None = None,  # the system turn: text, .txt, "none"; default per task
    guidance: float = 1.0,  # 1: the distilled recipe, one pass per step. Else real CFG against the empty prompt
    num_steps: int = 4,
    seed: int = 0,
    model_name: str = "flux.2-klein-9b",
    gpus=None,  # physical ids, 0,1,2,3. Default: CUDA_VISIBLE_DEVICES if set, else every GPU
    run_name: str | None = None,
    output_dir: str = "output",
    dry_run: bool = False,  # stop at the model's input (inpainting only: t2i has no input to show)
    toy: bool = False,  # random toy weights on the CPU: noise out, every line of code exercised
):
    bands = sorted({int(b) for b in as_list(bands)})
    assert bands and all(b >= 0 for b in bands), f"bands {bands}"
    assert "klein" in model_name, "the sampler here is the klein recipe (no guidance embedding)"
    prompt = read_text(prompt, FILL_PROMPT)

    if task == "t2i":
        assert prompts, "--prompts file.txt (one per line) or one prompt"
        texts = [line.strip() for line in Path(prompts).read_text().splitlines()] if str(prompts).endswith(".txt") else [str(prompts)]
        texts = [t for t in texts if t and not t.startswith("#")]
        assert size % 16 == 0 and max(bands) <= size // 16, f"size {size}, bands {bands}"
        assert not dry_run, "a dry run shows the model's input; text to image has none"
        jobs = [{"name": f"{k:02d}_{slug(text)}", "prompt": text} for k, text in enumerate(texts)]
        system_prompt = read_text(system_prompt, T2I_SYSTEM_PROMPT)
    else:
        assert task in EXPERIMENTS, f"{task} is not t2i or one of {list(EXPERIMENTS)}"
        tile_paths = find_tiles(tiles, only)
        for path in tile_paths:
            w, h = Image.open(path).size
            assert w % 16 == 0 and h % 16 == 0, f"{path} is {w}x{h}: run scripts/resize_tiles.py first"
            assert 0 < hole < min(w, h) and hole % 2 == 0, f"hole {hole} px in a {w}x{h} tile"
            assert max(bands) <= min(w, h) // 16, f"band {max(bands)} tokens is more than the tile is wide"
        if hole % 32:
            print(f"note: hole {hole}: its edges do not fall on the 16 px token grid")
        texts = [prompt]
        jobs = [{"name": Path(p).stem, "tile": p} for p in tile_paths]
        system_prompt = read_text(system_prompt, SYSTEM_PROMPT)

    run = Path(output_dir) / (run_name or f"{time.strftime('%m%d_%H%M')}_copy_{'t2i' if task == 't2i' else 'inpaint'}")
    assert not (run / "config.json").exists(), f"{run} already holds a run: pick another --run_name"
    (run / "logs").mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    config = {
        "run_name": run.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "commit": commit.stdout.strip() or None,
        "experiment": "4_copy_tokens",
        "task": task,
        "bands": bands,
        "compare_rope": bool(compare_rope),
        "hole": hole,
        "size": int(size),
        "system_prompt": system_prompt,
        "prompt": prompt,
        "prompts": texts,  # everything the text encoder has to encode
        "guidance": float(guidance),
        "num_steps": int(num_steps),
        "seed": int(seed),
        "model_name": model_name,
        "dry_run": bool(dry_run),
        "toy": bool(toy),
        "jobs": jobs,
    }
    (run / "config.json").write_text(json.dumps(config, indent=2))

    if dry_run or toy:  # CPU only: a few processes, no GPU claimed
        count = min(len(jobs), 2 if toy else max(1, (os.cpu_count() or 2) // 2), 8)
        slots = [(f"cpu{k}", "") for k in range(count)]
    else:
        slots = [(f"gpu{g}", g) for g in find_gpus(gpus)][: len(jobs)]
        fetch_weights(model_name)
    print(f"{run}: {len(jobs)} jobs x {len(bands) + bool(compare_rope)} generations on {len(slots)} workers")

    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))}
    workers = []
    for shard, (label, device) in enumerate(slots):
        log = (run / "logs" / f"{label}.log").open("w")
        command = [sys.executable, "-m", "seamless.copy_tokens", str(run), str(shard), str(len(slots))]
        workers.append(subprocess.Popen(command, env=env | {"CUDA_VISIBLE_DEVICES": device}, stdout=log, stderr=subprocess.STDOUT))
    print(f"follow one with:  tail -f {run}/logs/{slots[0][0]}.log")

    done = -1
    while any(worker.poll() is None for worker in workers):
        time.sleep(2)
        if (count := len(list(run.glob("*/run.json")))) != done:
            done = count
            print(f"  {done}/{len(jobs)} jobs", flush=True)
    done = len(list(run.glob("*/run.json")))
    failed = [shard for shard, worker in enumerate(workers) if worker.returncode != 0]

    index = write_report(str(run))
    print(f"{done}/{len(jobs)} jobs done   ->   {index}   and   {run}/summary.csv")
    if failed:
        sys.exit(f"workers {failed} failed: see {run}/logs/ (the report holds what finished)")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
