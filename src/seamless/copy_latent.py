"""Experiment 5: copy only at the input (flux2/copy_latent.py), text to image, as a sweep over its band.

Every prompt is generated several times from the same noise, the model and the 4-step recipe the same,
one column per way of making the picture seamless. The columns:

  stock     band 0: the stock sampler, the latent decoded around the torus. The baseline.
  latent b  the latent padded by b tokens of circular copies before every forward pass, the velocity
            of the middle chopped after it, the padded again. Stock attention, stock RoPE. The method.
  tokens b  (--controls tokens) experiment 4: the same b tokens, but as extra keys and values in
            every attention layer. Same noise, same positions of the copies, copying done elsewhere.
  rope      (--controls rope) experiments 1-3: the nearest-copy RoPE, no copies.

The noise is the one experiment 4 draws for the same seed and size, so a sheet of this experiment
and a sheet of experiment 4 can be laid side by side. Judged by the seam jump of the result (1 = the
seam is like any other line of the picture), and by eye on the 2x2 repeat.

    python -m seamless.copy_latent <run_dir> <shard> <num_shards>      one worker; the launcher
    python -m seamless.copy_latent report <run_dir>                    rebuild index.html and summary.csv

scripts/run_copy_latent.py starts the workers and says what comes out.
"""

import csv
import html
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

from flux2.copy_latent import denoise_copy_latent
from flux2.sampling import batched_prc_txt, get_schedule
from flux2.torus import decode_torus

from .copy_tokens import KleinCopy, slug  # noqa: F401  slug is re-exported for the launcher
from .report import STYLE, fmt
from .sheet import Panel, build_sheet
from .tiles import seam_jump, tiled


class KleinLatent(KleinCopy):
    """`KleinCopy` with one more way to draw: the copy at the input. The weights are loaded once for the sweep."""

    @torch.no_grad()
    def generate(self, prompt: str, seed: int, method: str, band: int | None, size: int) -> Image.Image:
        """`method` "latent", "tokens" or "rope"; `band` is ignored by "rope". Same seed, same noise."""
        if method == "tokens":
            return self(prompt, seed, band, None, size)
        if method == "rope":
            return self(prompt, seed, None, None, size)

        assert size % 16 == 0, f"{size} is not a multiple of 16"
        gh = gw = size // 16
        weight = next(self.model.parameters())
        device, dtype = weight.device, weight.dtype

        ctx = self.ctx[prompt] if self.guidance == 1 else torch.cat((self.ctx[""], self.ctx[prompt]))
        ctx, ctx_ids = batched_prc_txt(ctx.to(device, dtype))

        generator = torch.Generator(device=device).manual_seed(seed)  # the draw of KleinCopy.__call__
        core = torch.randn((1, 128, gh, gw), generator=generator, dtype=dtype, device=device)

        # The schedule of the core's token count, as in every other column: see flux2/copy_latent.py.
        timesteps = get_schedule(self.num_steps, gh * gw)
        core = denoise_copy_latent(self.model, core, ctx, ctx_ids, timesteps, band, self.guidance)

        x = decode_torus(self.ae, core).float().clamp(-1, 1)
        return Image.fromarray((127.5 * (rearrange(x[0], "c h w -> h w c") + 1.0)).cpu().byte().numpy())


def variants(config: dict) -> list[tuple[str, str, int | None]]:
    """(label, method, band) of every column of the sweep."""
    columns = [("stock" if b == 0 else f"latent{b}", "latent", b) for b in config["bands"]]
    if "tokens" in config["controls"]:  # tokens at band 0 is the stock column again
        columns += [(f"tokens{b}", "tokens", b) for b in config["bands"] if b > 0]
    if "rope" in config["controls"]:
        columns.append(("rope", "rope", None))
    return columns


def run_job(job: dict, run_dir: str | Path, config: dict, klein: KleinLatent):
    """One prompt ({"name", "prompt"}) through the whole sweep. Writes <run_dir>/<name>/: one PNG per
    panel, sheet.png, thumb.jpg, run.json."""
    folder = Path(run_dir) / job["name"]
    folder.mkdir(parents=True, exist_ok=True)
    record = job | {"variants": {}}
    panels = []
    for label, method, band in variants(config):
        started = time.time()
        image = klein.generate(job["prompt"], config["seed"], method, band, config["size"])
        seam = round(seam_jump(image), 2)
        record["variants"][label] = {"method": method, "band": band, "seconds": round(time.time() - started, 2), "seam": seam}
        panels += [
            Panel(label, f"{label}: generated", image, tag=f"seam {seam:.2f}"),
            Panel(f"{label}_2x2", f"{label}, 2x2", tiled(image), copies=2, marks="seam"),
        ]
        print(f"{job['name']} {label}  seam {seam:.2f}", flush=True)

    for k, panel in enumerate(panels, start=1):
        panel.name = f"{k}_{panel.name}"
        panel.image.save(folder / f"{panel.name}.png")
    seams = "   ".join(f"{label} {v['seam']:.2f}" for label, v in record["variants"].items())
    header = [
        f"Experiment 5: copy at the input, text to image   |   {job['name']}",
        f"The picture from the prompt alone, {config['size']} x {config['size']}, once per column from the same noise. "
        "stock: stock sampler. latent b: the latent padded by b tokens of circular copies before every forward pass, "
        "the velocity chopped to the middle after it. tokens b: b tokens of copies as keys and values in every layer "
        "(experiment 4). rope: nearest-copy RoPE (experiments 1-3).",
        f"{config['model_name']}   {config['num_steps']} steps   guidance {config['guidance']:g}   seed {config['seed']}   bands {config['bands']}",
        f"system: {config['system_prompt'] or '(none: bare user turn)'}",
        f"prompt: {job['prompt']}",
        f"seam jump (1 = like any other line):   {seams}",
    ]
    sheet = build_sheet(panels, 0, header)
    sheet.save(folder / "sheet.png", compress_level=3)
    thumb = sheet.copy()
    thumb.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    thumb.save(folder / "thumb.jpg", quality=85)
    (folder / "run.json").write_text(json.dumps(record, indent=2))  # last: its presence means "done"


def write_report(run_dir: str) -> Path:
    """index.html and summary.csv from the run.json of every job; can be rerun at any time."""
    run = Path(run_dir)
    config = json.loads((run / "config.json").read_text())
    records = {r["name"]: r for r in (json.loads(p.read_text()) for p in run.glob("*/run.json"))}
    labels = [label for label, _, _ in variants(config)]

    with (run / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_name", "variant", "method", "band", "seam", "seconds", "sheet"])
        for job in config["jobs"]:
            for label, v in (records[job["name"]]["variants"] if job["name"] in records else {}).items():
                writer.writerow([job["name"], label, v["method"], v["band"], v["seam"], v["seconds"], f"{job['name']}/sheet.png"])

    out = [f"<!doctype html><meta charset=utf-8><title>{html.escape(run.name)}</title><style>{STYLE}</style>"]
    out.append(f"<h1>{html.escape(run.name)}: experiment 5, copy at the input, text to image</h1>")
    out.append(
        f"<p>{html.escape(config['model_name'])}{' (TOY WEIGHTS: noise)' if config['toy'] else ''} &middot; "
        f"{config['num_steps']} steps &middot; guidance {config['guidance']:g} &middot; seed {config['seed']} &middot; "
        f"{config['size']} x {config['size']} &middot; bands <code>{config['bands']}</code> tokens &middot; "
        f"controls <code>{config['controls']}</code> &middot; {len(config['jobs'])} prompts &middot; commit {html.escape(str(config['commit']))}</p>"
        f"<p>system: {html.escape(config['system_prompt'] or '(none: bare user turn)')}</p>"
    )
    done = [r for r in records.values() if r["variants"]]
    if done:
        out.append("<h2>Median over prompts</h2><table><tr><th>variant</th><th>prompts</th><th>seam jump</th><th>seconds</th></tr>")
        for label in labels:
            rows = [r["variants"][label] for r in done if label in r["variants"]]
            if rows:
                out.append(
                    f"<tr><td>{label}</td><td class=num>{len(rows)}</td><td class=num>{fmt(statistics.median(v['seam'] for v in rows))}</td>"
                    f"<td class=num>{fmt(statistics.median(v['seconds'] for v in rows), 1)}</td></tr>"
                )
        out.append("</table><dl><dt><code>seam jump</code></dt><dd>how much stronger the colour jump across the picture's own seam is than across any other line (1 = no seam)</dd></dl>")

    out.append("<h2>Sheets</h2><p>Click one for the full-resolution sheet; the folder link has every panel as its own PNG.</p>")
    out.append("<table class=grid><tr><th>prompt</th><th>sheet</th><th>seam jump per variant</th></tr>")
    for job in config["jobs"]:
        r, name = records.get(job["name"]), job["name"]
        out.append(f"<tr><td class=name>{html.escape(name)}<br><span class=m>{html.escape(job['prompt'])}</span></td>")
        if r is None:
            out.append("<td class=m colspan=2>missing: see logs/</td></tr>")
            continue
        out.append(f"<td><a href='{name}/sheet.png'><img loading=lazy src='{name}/thumb.jpg'></a><a href='{name}/'>folder</a></td>")
        out.append("<td class=m>" + "<br>".join(f"{label}: {v['seam']:.2f}" for label, v in r["variants"].items()) + "</td></tr>")
    out.append("</table>")
    (run / "index.html").write_text("\n".join(out))
    return run / "index.html"


def main(run_dir: str, shard: int = 0, num_shards: int = 1):
    """One process, one GPU: the sweep on jobs[shard::num_shards]. config.json in the run folder says the rest."""
    config = json.loads((Path(run_dir) / "config.json").read_text())
    jobs = config["jobs"][shard::num_shards]
    print(f"shard {shard} of {num_shards}: {len(jobs)} jobs", flush=True)

    settings = {k: config[k] for k in ("num_steps", "guidance")}
    load = KleinLatent.toy if config["toy"] else lambda *a, **kw: KleinLatent.load(config["model_name"], *a, **kw)
    klein = load(config["prompts"], config["system_prompt"], **settings)

    failed = []
    for job in jobs:
        try:
            run_job(job, run_dir, config, klein)
        except Exception:  # noqa: BLE001  one bad job must not cost the others their GPU time
            traceback.print_exc()
            failed.append(job["name"])
    if failed:
        sys.exit(f"shard {shard}: failed on {failed}")
    print(f"shard {shard}: done", flush=True)


if __name__ == "__main__":
    from fire import Fire

    Fire({"worker": main, "report": write_report} if sys.argv[1:2] == ["report"] else main)
