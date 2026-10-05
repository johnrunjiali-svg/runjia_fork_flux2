"""One process, one GPU: every experiment of a run on its share of the tiles.

    python -m seamless.worker <run_dir> <shard> <num_shards>

scripts/run_experiments.py starts one of these per GPU with CUDA_VISIBLE_DEVICES set; there is no
reason to start one by hand. All it is told is the run folder: config.json in it says the rest.
The share is tiles[shard::num_shards], whole tiles, so that two experiments which hand the model
the same picture of a tile meet in one process and the picture is generated once.
"""

import json
import sys
import traceback
from pathlib import Path

from .experiments import run_tile


def main(run_dir: str, shard: int = 0, num_shards: int = 1):
    config = json.loads((Path(run_dir) / "config.json").read_text())
    tiles = config["tiles"][shard::num_shards]
    print(f"shard {shard} of {num_shards}: {len(tiles)} tiles", flush=True)

    generate = None
    if not config["dry_run"]:
        from .klein import Klein  # torch, and minutes of loading: not for a dry run

        settings = {k: config[k] for k in ("num_steps", "guidance", "wrap", "unanchor_text")}
        if config["toy"]:
            klein = Klein.toy([config["prompt"]], config["system_prompt"], **settings)
        else:
            klein = Klein.load(config["model_name"], [config["prompt"]], config["system_prompt"], **settings)

        def generate(reference, prompt):
            return klein(reference, prompt, seed=config["seed"])

    failed = []
    for tile in tiles:
        try:
            run_tile(tile, run_dir, config, generate)
        except Exception:  # noqa: BLE001  one bad tile must not cost the others their GPU time
            traceback.print_exc()
            failed.append(tile)
    if failed:
        sys.exit(f"shard {shard}: failed on {failed}")
    print(f"shard {shard}: done", flush=True)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
