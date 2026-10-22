"""Experiment 5, on the server: copy only at the input, text to image, swept over the band. One process per GPU.

Each forward pass gets the latent padded by `band` tokens of circular copies; the velocity is chopped
back to the middle, the Euler step moves the middle, the next pass pads it again (flux2/copy_latent.py).

    uv run python scripts/run_copy_latent.py                                   the prompts of principled/t2i_objects.txt, bands 0, 2, 4, 8, 16
    uv run python scripts/run_copy_latent.py --bands 0,1,2,4,8,16,32           the sweep; each band is one more generation per prompt
    uv run python scripts/run_copy_latent.py --controls tokens,rope            add experiment 4's copy tokens (same bands) and experiments 1-3's RoPE as columns
    uv run python scripts/run_copy_latent.py --controls none                   latent columns only
    uv run python scripts/run_copy_latent.py --prompts "red roses on green vines"      one prompt; or --prompts file.txt, one per line (blank and # lines skipped)
    uv run python scripts/run_copy_latent.py --only 0,3,rose --gpus 0,1        of the prompts, the ones whose number or words match; on these GPUs
    uv run python scripts/run_copy_latent.py --size 1024 --seed 1 --system_prompt none

    uv run python scripts/run_copy_latent.py --toy --only 0,1 --bands 0,1      random toy weights on the CPU: the plumbing

The system turn is T2I_SYSTEM_PROMPT of experiment 4 unless --system_prompt text|file.txt|none.
Same seed and size as an experiment 4 run: same noise, so its sheets and these compare column by column.

What the experiment is is in src/seamless/copy_latent.py; the method in src/flux2/copy_latent.py.

    output/<run_name>/
        index.html, summary.csv         median seam jump per column, and every prompt with its numbers
        config.json, logs/gpu<k>.log
        <k>_<first words of the prompt>/
            1_*.png ...                 every panel as its own PNG: result and 2x2 per column
            sheet.png, thumb.jpg        all of them side by side at full resolution
            run.json
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_copy_tokens import read_text
from run_experiments import REPO, as_list, fetch_weights, find_gpus
from seamless.copy_latent import slug, write_report
from seamless.copy_tokens import T2I_SYSTEM_PROMPT


def main(
    prompts: str = "principled/t2i_objects.txt",  # a .txt with one prompt per line, or one prompt
    only=None,  # of those, the ones whose line number (from 0) or words match: 0,3,rose
    size: int = 1024,  # the square picture
    bands="0,2,4,8,16",  # padding in tokens (16 px each); 0 is the stock sampler
    controls="rope",  # more columns: tokens (experiment 4, at the same bands), rope (experiments 1-3), both as tokens,rope, or none
    system_prompt: str | None = None,  # the system turn: text, .txt, "none"; default T2I_SYSTEM_PROMPT
    guidance: float = 1.0,  # 1: the distilled recipe, one pass per step. Else real CFG against the empty prompt
    num_steps: int = 4,
    seed: int = 0,
    model_name: str = "flux.2-klein-9b",
    gpus=None,  # physical ids, 0,1,2,3. Default: CUDA_VISIBLE_DEVICES if set, else every GPU
    run_name: str | None = None,
    output_dir: str = "output",
    toy: bool = False,  # random toy weights on the CPU: noise out, every line of code exercised
):
    bands = sorted({int(b) for b in as_list(bands)})
    controls = [c for c in as_list(controls) if c != "none"]
    assert bands and all(0 <= b <= size // 16 for b in bands) and size % 16 == 0, f"size {size}, bands {bands}"
    assert set(controls) <= {"tokens", "rope"}, f"controls {controls}: tokens, rope or none"
    assert "klein" in model_name, "the sampler here is the klein recipe (no guidance embedding)"

    path = Path(prompts)
    texts = [line.strip() for line in path.read_text().splitlines()] if path.suffix == ".txt" else [str(prompts)]
    texts = [t for t in texts if t and not t.startswith("#")]
    jobs = [{"name": f"{k:02d}_{slug(text)}", "prompt": text} for k, text in enumerate(texts)]
    if keys := as_list(only):
        jobs = [j for j in jobs if any(int(k) == int(j["name"][:2]) if k.isdigit() else k.lower() in j["prompt"].lower() for k in keys)]
    assert jobs, f"no prompt in {prompts} matches {only}"
    system_prompt = read_text(system_prompt, T2I_SYSTEM_PROMPT)

    run = Path(output_dir) / (run_name or f"{time.strftime('%m%d_%H%M')}_copy_latent")
    assert not (run / "config.json").exists(), f"{run} already holds a run: pick another --run_name"
    (run / "logs").mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    config = {
        "run_name": run.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "commit": commit.stdout.strip() or None,
        "experiment": "5_copy_latent",
        "bands": bands,
        "controls": controls,
        "size": int(size),
        "system_prompt": system_prompt,
        "prompts": [j["prompt"] for j in jobs],  # everything the text encoder has to encode
        "guidance": float(guidance),
        "num_steps": int(num_steps),
        "seed": int(seed),
        "model_name": model_name,
        "toy": bool(toy),
        "jobs": jobs,
    }
    (run / "config.json").write_text(json.dumps(config, indent=2))

    if toy:  # CPU only: a couple of processes, no GPU claimed
        slots = [(f"cpu{k}", "") for k in range(min(len(jobs), 2))]
    else:
        slots = [(f"gpu{g}", g) for g in find_gpus(gpus)][: len(jobs)]
        fetch_weights(model_name)
    columns = len(bands) + len([b for b in bands if b > 0]) * ("tokens" in controls) + ("rope" in controls)
    print(f"{run}: {len(jobs)} prompts x {columns} generations on {len(slots)} workers")

    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))}
    workers = []
    for shard, (label, device) in enumerate(slots):
        log = (run / "logs" / f"{label}.log").open("w")
        command = [sys.executable, "-m", "seamless.copy_latent", str(run), str(shard), str(len(slots))]
        workers.append(subprocess.Popen(command, env=env | {"CUDA_VISIBLE_DEVICES": device}, stdout=log, stderr=subprocess.STDOUT))
    print(f"follow one with:  tail -f {run}/logs/{slots[0][0]}.log")

    done = -1
    while any(worker.poll() is None for worker in workers):
        time.sleep(2)
        if (count := len(list(run.glob("*/run.json")))) != done:
            done = count
            print(f"  {done}/{len(jobs)} prompts", flush=True)
    done = len(list(run.glob("*/run.json")))
    failed = [shard for shard, worker in enumerate(workers) if worker.returncode != 0]

    index = write_report(str(run))
    print(f"{done}/{len(jobs)} prompts done   ->   {index}   and   {run}/summary.csv")
    if failed:
        sys.exit(f"workers {failed} failed: see {run}/logs/ (the report holds what finished)")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
