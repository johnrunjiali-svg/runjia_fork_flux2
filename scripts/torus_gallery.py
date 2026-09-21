"""Merge the manifest_*.jsonl of a run into <run>/gallery.md:  python scripts/torus_gallery.py output/<run_name>"""

import json
import sys
from pathlib import Path


def main(run_dir: str):
    run = Path(run_dir)
    records = [json.loads(line) for f in sorted(run.glob("manifest_*.jsonl")) for line in f.open()]
    config = json.loads((run / "config.json").read_text())

    shown = [
        "model_name",
        "width",
        "height",
        "num_steps",
        "guidance",
        "geo_guidance",
        "wrap_h",
        "wrap_w",
        "unanchor_text",
    ]
    md = [f"# {run.name}", "", ", ".join(f"`{k}={config[k]}`" for k in shown), ""]
    md += [
        f"{len(records)} images. Each cell: the image, then its 2x2 tiling (seams meet in the middle).",
        "",
    ]
    md += [f"Geometry sentence: _{config['geo_prompt']}_", ""]

    for set_name in sorted({r["set"] for r in records}):
        md += [f"## {set_name}", ""]
        in_set = [r for r in records if r["set"] == set_name]
        for name in sorted({r["name"] for r in in_set}):
            rows = sorted((r for r in in_set if r["name"] == name), key=lambda r: r["seed"])
            md += [f"### {name}", "", f"> {rows[0]['prompt']}", ""]
            md += ["| " + " | ".join(f"seed {r['seed']}" for r in rows) + " |", "|" + " --- |" * len(rows)]
            for suffix in ("", "_tiled"):
                cells = [f"![]({set_name}/{name}/seed{r['seed']}{suffix}.png)" for r in rows]
                md += ["| " + " | ".join(cells) + " |"]
            md += [""]

    (run / "gallery.md").write_text("\n".join(md))
    print(f"Wrote {run / 'gallery.md'} ({len(records)} images)")


if __name__ == "__main__":
    main(sys.argv[1])
