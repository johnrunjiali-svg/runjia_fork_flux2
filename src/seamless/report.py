"""A finished run as one table: summary.csv for a spreadsheet, index.html for the eye.

Both are rebuilt from the run.json of every job, so they can be made again at any time, also for a
run that is still going or lost a shard:    python -m seamless.report output/<run>
"""

import csv
import html
import json
import statistics
from pathlib import Path

from .experiments import rope_label

METRICS = {
    "seam_before": "seam jump of the tile the experiment starts from (1 = like any other line)",
    "seam_after": "seam jump of the result",
    "wrap_after": "jump where the generated picture's own edges meet, after rolling back",
    "kept_psnr": "dB between the model's input and output outside the hole",
}

STYLE = """
body { background:#18181b; color:#ececec; font:14px/1.45 system-ui,sans-serif; margin:24px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:26px 0 8px; }
p, dd { color:#a0a0a5; margin:2px 0; max-width:1100px; } dt { margin-top:8px; } code { color:#ffc428; }
table { border-collapse:collapse; } th, td { text-align:left; padding:6px 14px 6px 0; vertical-align:top; }
th { color:#a0a0a5; font-weight:500; border-bottom:1px solid #3f3f46; }
td.num { font-variant-numeric:tabular-nums; }
.grid td { padding:10px 14px 18px 0; border-bottom:1px solid #27272a; }
.grid img { display:block; width:440px; border:1px solid #3f3f46; margin-bottom:5px; }
a { color:#46aaff; text-decoration:none; } a:hover { text-decoration:underline; }
.m { color:#a0a0a5; font-variant-numeric:tabular-nums; } .name { width:150px; overflow-wrap:anywhere; }
"""


def load(run: Path) -> tuple[dict, list[dict]]:
    config = json.loads((run / "config.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted(run.glob("*/*/run.json"))]
    return config, records


def fmt(value: float | None, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_report(run_dir: str) -> Path:
    run = Path(run_dir)
    config, records = load(run)
    measured = [r for r in records if "metrics" in r]

    with (run / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "tile", "hole", *METRICS, "seconds", "same_generation_as", "sheet"])
        for r in measured:
            sheet = f"{r['experiment']}/{r['tile']}/sheet.png"
            values = [r["metrics"][m] for m in METRICS]
            writer.writerow(
                [
                    r["experiment"],
                    r["tile"],
                    r["hole"],
                    *values,
                    r["seconds"],
                    r["same_generation_as"] or "",
                    sheet,
                ]
            )

    out = [f"<!doctype html><meta charset=utf-8><title>{html.escape(run.name)}</title><style>{STYLE}</style>"]
    out.append(f"<h1>{html.escape(run.name)}</h1>")
    if config["dry_run"]:
        out.append("<p>Dry run: the model's inputs only, nothing was generated.</p>")
    else:
        out.append(
            f"<p>{html.escape(config['model_name'])}{' (TOY WEIGHTS: noise)' if config['toy'] else ''} &middot; "
            f"{config['num_steps']} steps &middot; guidance {config['guidance']:g} &middot; "
            f"seamless RoPE {html.escape(rope_label(config))} &middot; seed {config['seed']}</p>"
        )
    out.append(
        f"<p>cross <code>--band {config['band']}</code> px &middot; frame <code>--border {config['border']}</code> px"
        f" &middot; {len(config['tiles'])} tiles &middot; commit {html.escape(str(config['commit']))}</p>"
        f"<p>system: {html.escape(config.get('system_prompt') or '(none: bare user turn)')}</p>"
        f"<p>prompt: {html.escape(config['prompt'])}</p>"
    )

    if measured:
        out.append("<h2>Median over tiles</h2><table><tr><th>experiment</th><th>tiles</th>")
        out += [f"<th>{m}</th>" for m in METRICS]
        out.append("</tr>")
        for name in config["experiments"]:
            rows = [r["metrics"] for r in measured if r["experiment"] == name]
            out.append(f"<tr><td>{name}</td><td class=num>{len(rows)}</td>")
            for m in METRICS:
                values = [row[m] for row in rows if row[m] is not None]
                out.append(f"<td class=num>{fmt(statistics.median(values)) if values else ''}</td>")
            out.append("</tr>")
        out.append("</table><dl>")
        out += [f"<dt><code>{m}</code></dt><dd>{html.escape(text)}</dd>" for m, text in METRICS.items()]
        out.append("</dl>")

    by_job = {(r["experiment"], r["tile"]): r for r in records}
    out.append(
        "<h2>Sheets</h2><p>Click one for the full-resolution sheet; the folder link has every step as its own PNG.</p>"
    )
    out.append(
        "<table class=grid><tr><th>tile</th>"
        + "".join(f"<th>{n}</th>" for n in config["experiments"])
        + "</tr>"
    )
    for tile in [Path(t).stem for t in config["tiles"]]:
        out.append(f"<tr><td class=name>{html.escape(tile)}</td>")
        for name in config["experiments"]:
            r = by_job.get((name, tile))
            if r is None:
                out.append("<td class=m>missing: see logs/</td>")
                continue
            folder = f"{name}/{tile}"
            line = ""
            if "metrics" in r:
                m = r["metrics"]
                line = f"seam {fmt(m['seam_before'])} &rarr; {fmt(m['seam_after'])}"
                line += f" &middot; wrap {fmt(m['wrap_after'])}" if m["wrap_after"] is not None else ""
                line += f" &middot; kept {fmt(m['kept_psnr'], 1)} dB"
                if r["same_generation_as"]:
                    line += f"<br>same generation as {r['same_generation_as']}"
            out.append(
                f"<td><a href='{folder}/sheet.png'><img loading=lazy src='{folder}/thumb.jpg'></a>"
                f"<span class=m>{line}</span> <a href='{folder}/'>folder</a></td>"
            )
        out.append("</tr>")
    out.append("</table>")
    (run / "index.html").write_text("\n".join(out))
    return run / "index.html"


if __name__ == "__main__":
    from fire import Fire

    Fire(write_report)
