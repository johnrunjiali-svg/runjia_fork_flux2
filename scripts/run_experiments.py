"""Step 1, on the server: run the experiments on a set of tiles, one process per GPU.

    uv run python scripts/run_experiments.py                              all tiles, all three experiments, all GPUs
    uv run python scripts/run_experiments.py --only 00,03,19              the tiles whose number is 0, 3 or 19
    uv run python scripts/run_experiments.py --only rose,ivy              ... or whose name contains one of these
    uv run python scripts/run_experiments.py --tiles a.png,more/tiles     any files, folders or globs
    uv run python scripts/run_experiments.py --experiments 1              1, 2, 3 or their full names
    uv run python scripts/run_experiments.py --band 64                    width of the white cross (and frame = band / 2)
    uv run python scripts/run_experiments.py --band 128 --border 32       ... or set the frame of 2 and 3 on its own
    uv run python scripts/run_experiments.py --prompt "Fill the white."   or --prompt prompt.txt
    uv run python scripts/run_experiments.py --system_prompt none         no system turn: the bare prompt (leaves holes unfilled)
    uv run python scripts/run_experiments.py --guidance 2.5               real CFG, two passes per step instead of one
    uv run python scripts/run_experiments.py --wrap False                 stock attention: the baseline without seamless RoPE
    uv run python scripts/run_experiments.py --rope quantized             frequencies rounded to the grid instead of the nearest copy
    uv run python scripts/run_experiments.py --rope configs/rope/r7k9.json ... plane by plane, as chosen in scripts/rope_frequencies.py
    uv run python scripts/run_experiments.py --gpus 0,1,2,3 --run_name first

    uv run python scripts/run_experiments.py --dry_run                    no model: the masks and inputs only (laptop)
    uv run python scripts/run_experiments.py --toy --only 0,1             random toy weights on the CPU: the plumbing

What the experiments are is in src/seamless/experiments.py. What comes out:

    output/<run_name>/
        index.html, summary.csv         every job in one table; open the first, sort the second
        config.json                     everything needed to repeat the run
        logs/gpu<k>.log
        <experiment>/<tile>/
            1_original.png ... N_*.png  every step as its own PNG, numbered in the order it was made
            sheet.png                   all of them side by side at full resolution, seams marked
            thumb.jpg, run.json

To look at a run from the laptop, either copy it (rsync -av <server>:<repo>/output/<run> .) or serve it:
    python -m http.server 8000 --bind 127.0.0.1 --directory output/<run>      then  ssh -L 8000:localhost:8000 <server>
"""

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from seamless.experiments import EXPERIMENTS, FILL_PROMPT, SYSTEM_PROMPT
from seamless.report import write_report

REPO = Path(__file__).resolve().parents[1]
IMAGES = {".png", ".jpg", ".jpeg", ".webp"}


def as_list(value) -> list[str]:
    """What fire hands over for `a,b`, `3` or `(0, 3)`, as a list of strings."""
    if value is None:
        return []
    items = (
        value.split(",") if isinstance(value, str) else value if isinstance(value, (list, tuple)) else [value]
    )
    return [str(item).strip() for item in items if str(item).strip()]


def find_tiles(tiles, only) -> list[str]:
    paths: list[Path] = []
    for item in as_list(tiles):
        if Path(item).is_dir():
            paths += sorted(p for p in Path(item).iterdir() if p.suffix.lower() in IMAGES)
        else:
            found = sorted(Path(p) for p in glob.glob(item))
            assert found, f"no such tile: {item}"
            paths += found

    keys = as_list(only)

    def wanted(path: Path) -> bool:
        number = path.stem.split("_")[0]
        return any(
            int(number) == int(key)
            if key.isdigit() and number.isdigit()
            else key.lower() in path.stem.lower()
            for key in keys
        )

    paths = [p for p in paths if wanted(p)] if keys else paths
    assert paths, f"no tiles in {tiles}" + (f" match {keys}" if keys else "")
    assert len({p.stem for p in paths}) == len(
        paths
    ), "two tiles share a file name; their folders would collide"
    return [p.as_posix() for p in paths]


def find_experiments(experiments) -> list[str]:
    keys = as_list(experiments)
    names = [n for n in EXPERIMENTS if not keys or n in keys or n.split("_")[0] in keys]
    assert names and (not keys or len(names) == len(set(keys))), f"{keys} are not among {list(EXPERIMENTS)}"
    return names


def find_gpus(gpus) -> list[str]:
    """The physical GPU ids to use, one worker each: --gpus, else CUDA_VISIBLE_DEVICES, else all."""
    if as_list(gpus):
        return as_list(gpus)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return as_list(os.environ["CUDA_VISIBLE_DEVICES"])
    import torch

    assert (
        torch.cuda.is_available()
    ), "no GPU here: --dry_run shows the inputs, --toy runs toy weights on the CPU"
    return [str(k) for k in range(torch.cuda.device_count())]


def fetch_weights(model_name: str):
    """Fill the Hugging Face cache once, so eight processes do not download the same files at once."""
    from huggingface_hub import hf_hub_download, snapshot_download

    from flux2.util import FLUX2_MODEL_INFO

    info = FLUX2_MODEL_INFO[model_name]
    snapshot_download(f"Qwen/Qwen3-{'4B' if '4b' in model_name else '8B'}-FP8")
    if info["model_path"] not in os.environ:
        hf_hub_download(info["repo_id"], info["filename"])
    if "AE_MODEL_PATH" not in os.environ:
        hf_hub_download(info.get("ae_repo_id", info["repo_id"]), info["filename_ae"])


def main(
    tiles: str = "principled/tiles_1024",  # folders, files or globs, comma separated
    only=None,  # of those, the ones whose leading number or name matches: 00,03,19 or rose,ivy
    experiments=None,  # default: all three. 1 / 1,3 / 2_outpaint_frame
    band: int = 128,  # experiment 1: width of the white cross over the seams
    border: int
    | None = None,  # experiments 2 and 3: width of the frame cut off and painted white. Default band / 2
    prompt: str = FILL_PROMPT,  # the instruction, or a .txt file that holds it
    system_prompt: str = SYSTEM_PROMPT,  # the system turn in front of it (text or .txt). "none": no system turn
    guidance: float = 1.0,  # 1: the distilled recipe, one pass per step. Else real CFG against the empty prompt
    num_steps: int = 4,
    seed: int = 0,
    wrap: bool = True,  # the seamless RoPE and the circular decode. False: stock FLUX.2
    rope: str = "nearest",  # how the RoPE is made periodic, see flux2/torus.py: "nearest" copy, "quantized" frequencies, or a rules .json from scripts/rope_frequencies.py
    unanchor_text: bool = False,  # see flux2/torus.py: the text no longer marks an origin on the torus
    model_name: str = "flux.2-klein-9b",
    gpus=None,  # physical ids, 0,1,2,3. Default: CUDA_VISIBLE_DEVICES if set, else every GPU
    run_name: str | None = None,
    output_dir: str = "output",
    dry_run: bool = False,  # stop at the model's input: the masks, without a model
    toy: bool = False,  # random toy weights on the CPU: noise out, every line of code exercised
):
    tile_paths = find_tiles(tiles, only)
    names = find_experiments(experiments)
    border = band // 2 if border is None else border
    prompt = Path(prompt).read_text().strip() if str(prompt).endswith(".txt") else str(prompt)
    system_prompt = (
        "" if system_prompt is None or str(system_prompt).lower() == "none" else str(system_prompt)
    )
    system_prompt = (
        Path(system_prompt).read_text().strip() if system_prompt.endswith(".txt") else system_prompt
    )
    assert "klein" in model_name, "the sampler here is the klein recipe (no guidance embedding)"
    rules = None  # quantized with no rules: every plane rounded
    if rope.endswith(".json"):
        rules, rope = json.loads(Path(rope).read_text())["rules"], "quantized"
    assert rope in ("nearest", "quantized"), rope

    for path in tile_paths:
        w, h = Image.open(path).size
        assert w % 16 == 0 and h % 16 == 0, f"{path} is {w}x{h}: run scripts/resize_tiles.py first"
        holes = {band if not EXPERIMENTS[n].crop else 2 * border for n in names}
        assert all(
            0 < hole < min(w, h) and hole % 2 == 0 for hole in holes
        ), f"hole {holes} px in a {w}x{h} tile"
    if band % 32 or border % 16:
        print(f"note: band {band} / border {border}: the hole's edges do not fall on the 16 px token grid")

    run = Path(output_dir) / (run_name or f"{time.strftime('%m%d_%H%M')}_band{band}")
    assert not (run / "config.json").exists(), f"{run} already holds a run: pick another --run_name"
    (run / "logs").mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    config = {
        "run_name": run.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "commit": commit.stdout.strip() or None,
        "experiments": names,
        "band": band,
        "border": border,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "guidance": float(guidance),
        "num_steps": int(num_steps),
        "seed": int(seed),
        "wrap": bool(wrap),
        "rope": rope,
        "rules": rules,
        "unanchor_text": bool(unanchor_text),
        "model_name": model_name,
        "dry_run": bool(dry_run),
        "toy": bool(toy),
        "tiles": tile_paths,
    }
    (run / "config.json").write_text(json.dumps(config, indent=2))

    if dry_run or toy:  # CPU only: a few processes, no GPU claimed
        count = min(len(tile_paths), 2 if toy else max(1, (os.cpu_count() or 2) // 2), 8)
        slots = [(f"cpu{k}", "") for k in range(count)]
    else:
        slots = [(f"gpu{g}", g) for g in find_gpus(gpus)][: len(tile_paths)]
        fetch_weights(model_name)
    total = len(tile_paths) * len(names)
    print(f"{run}: {len(tile_paths)} tiles x {len(names)} experiments on {len(slots)} workers")

    workers = []
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))
    }
    for shard, (label, device) in enumerate(slots):
        log = (run / "logs" / f"{label}.log").open("w")
        command = [sys.executable, "-m", "seamless.worker", str(run), str(shard), str(len(slots))]
        workers.append(
            subprocess.Popen(
                command, env=env | {"CUDA_VISIBLE_DEVICES": device}, stdout=log, stderr=subprocess.STDOUT
            )
        )
    print(f"follow one with:  tail -f {run}/logs/{slots[0][0]}.log")

    done = -1
    while any(worker.poll() is None for worker in workers):
        time.sleep(2)
        if (count := len(list(run.glob("*/*/run.json")))) != done:
            done = count
            print(f"  {done}/{total} jobs", flush=True)
    done = len(list(run.glob("*/*/run.json")))
    failed = [shard for shard, worker in enumerate(workers) if worker.returncode != 0]

    index = write_report(str(run))
    print(f"{done}/{total} jobs done   ->   {index}   and   {run}/summary.csv")
    if failed:
        sys.exit(f"workers {failed} failed: see {run}/logs/ (the report holds what finished)")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
