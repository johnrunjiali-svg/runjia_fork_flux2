"""Experiment 4: the copy-token attention (flux2/copy_tokens.py) on two tasks, as a sweep over its band.

Every job -- a tile to repair or a prompt to draw -- is generated several times from the same noise:
once per copy `band`, plus once with the nearest-copy RoPE of experiments 1-3 as the control. Between
the columns of one sheet only the attention differs:

  band 0   stock attention (the decoder is handed the latent wrapped for every column alike)
  band b   every image query also sees the b rows and columns past each edge, as copies
  rope     flux2/torus.py: no copies, every pair at its nearest periodic displacement

The two tasks:

  inpaint   experiments.py's task 3 by default (--task picks 1 or 2 instead): the centre crop of a
            tile on a white canvas, rolled by half so the hole is a cross through the middle; the fill
            prompt and inpainter system turn of experiments 1-3; measured like them (experiments.measure).
  t2i       the picture from the prompt alone, nothing in view, `size` square. The system turn is
            T2I_SYSTEM_PROMPT below, a draft to edit. Measured by the seam jump of the result.

    python -m seamless.copy_tokens <run_dir> <shard> <num_shards>      one worker; the launcher
    python -m seamless.copy_tokens report <run_dir>                    rebuild index.html and summary.csv

scripts/run_copy_tokens.py starts the workers and says what comes out.
"""

import csv
import html
import json
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

from flux2.copy_tokens import build_copy_geometry, denoise_copy
from flux2.sampling import batched_prc_img, batched_prc_txt, default_images_prep, get_schedule, prc_img
from flux2.torus import build_torus_geometry, decode_torus, denoise_torus

from .experiments import EXPERIMENTS, after, before, measure
from .klein import Klein
from .report import METRICS, STYLE, fmt
from .sheet import Panel, build_sheet
from .tiles import seam_jump, tiled

# A draft. The klein models were conditioned on a bare user turn; the system turn is where to say what
# kind of picture every prompt of the set asks for, without editing the prompts. Edit here, or pass
# --system_prompt text|file.txt ("none" for the bare user turn).
T2I_SYSTEM_PROMPT = (
    "You are a text-to-image model that generates seamless, tileable textures and patterns. "
    "Every image you make is one period of an endless repeat: its left edge continues into its right "
    "edge and its top edge into its bottom edge, so that copies placed side by side join without any "
    "visible seam. Fill the whole frame evenly with the pattern; no border, frame, vignette, "
    "centred object or text."
)


class KleinCopy(Klein):
    """`Klein` with the attention chosen per call; the weights are loaded once for the whole sweep."""

    def __init__(self, model, ae, ctx, num_steps: int = 4, guidance: float = 1.0):
        super().__init__(model, ae, ctx, num_steps, guidance)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        seed: int = 0,
        band: int | None = 4,  # None: the control, the nearest-copy RoPE of flux2/torus.py
        reference: Image.Image | None = None,  # inpainting: the picture with the white hole, seen as a reference image
        size: int = 1024,  # text to image: the square picture, when there is no reference to take the size from
    ) -> Image.Image:
        """Klein.__call__ with the geometry and sampler chosen by `band`. Same seed, same noise, so
        one column of the sweep differs from the next in the attention and nothing else."""
        width, height = reference.size if reference is not None else (size, size)
        assert width % 16 == 0 and height % 16 == 0, f"{width}x{height} is not a multiple of 16"
        gh, gw = height // 16, width // 16
        weight = next(self.model.parameters())
        device, dtype = weight.device, weight.dtype

        ctx = self.ctx[prompt] if self.guidance == 1 else torch.cat((self.ctx[""], self.ctx[prompt]))
        ctx, ctx_ids = batched_prc_txt(ctx.to(device, dtype))

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((1, 128, gh, gw), generator=generator, dtype=dtype, device=device)
        x, x_ids = batched_prc_img(noise)

        ref = ref_ids = None
        if reference is not None:  # one token per 16 x 16 px, so the hole's edges land on the output's grid; t = 10 marks a reference
            pixels = default_images_prep(reference.convert("RGB"))[None].to(device, next(self.ae.parameters()).dtype)
            ref, ref_ids = prc_img(self.ae.encode(pixels)[0].to(dtype), t_coord=torch.tensor([10]))
            ref, ref_ids = ref[None], ref_ids[None]

        timesteps = get_schedule(self.num_steps, x.shape[1])
        if band is None:
            geo = build_torus_geometry(self.model, x_ids, ctx_ids, (gh, gw), ref_ids=ref_ids)
            x = denoise_torus(self.model, x, ctx, geo, timesteps, self.guidance, ref=ref)
        else:
            geo = build_copy_geometry(self.model, x_ids, ctx_ids, (gh, gw), band, ref_ids)
            x = denoise_copy(self.model, x, ctx, geo, timesteps, self.guidance, ref=ref)

        x = rearrange(x, "b (h w) c -> b c h w", h=gh, w=gw)
        x = decode_torus(self.ae, x).float().clamp(-1, 1)
        return Image.fromarray((127.5 * (rearrange(x[0], "c h w -> h w c") + 1.0)).cpu().byte().numpy())


def variants(config: dict) -> list[tuple[str, int | None]]:
    """(label, band) of every column of the sweep; band None is the RoPE control."""
    return [(f"band{b}", b) for b in config["bands"]] + ([("rope", None)] if config["compare_rope"] else [])


def slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40]


def run_job(job: dict, run_dir: str | Path, config: dict, klein: KleinCopy | None):
    """One tile ({"name", "tile"}) or one prompt ({"name", "prompt"}) through the whole sweep. Writes
    <run_dir>/<name>/: one PNG per panel, sheet.png, thumb.jpg, run.json. `klein` None is a dry run
    that stops at the model's input."""
    folder = Path(run_dir) / job["name"]
    folder.mkdir(parents=True, exist_ok=True)
    record = job | {"task": config["task"], "dry_run": klein is None, "variants": {}}
    if "tile" in job:
        exp, hole = EXPERIMENTS[config["task"]], config["hole"]
        inputs, half = before(exp, Image.open(job["tile"]).convert("RGB"), hole)
        record |= {"hole": hole, "roll": list(half)}
        prompt, reference = config["prompt"], inputs[-1].image
    else:
        inputs, half, hole = [], (0, 0), 0
        prompt, reference = job["prompt"], None
    panels = list(inputs)

    for label, band in variants(config) if klein is not None else []:
        started = time.time()
        generated = klein(prompt, config["seed"], band, reference, config["size"])
        if reference is not None:
            results = after(exp, generated, half)
            metrics = measure(exp, inputs + results, hole, half)
            result, repeat = next(p for p in results if p.tag == "result"), results[-1]
        else:
            metrics = {"seam_after": round(seam_jump(generated), 2)}
            result, repeat = Panel(label, "generated", generated), Panel(label, "2x2", tiled(generated), copies=2, marks="seam")
        record["variants"][label] = {"band": band, "seconds": round(time.time() - started, 2), "metrics": metrics}
        # Onto the sheet go the result and its 2x2 repeat, labelled with their numbers.
        result.name, result.title, result.tag = label, f"{label}: {result.title}", tagline(metrics)
        repeat.name, repeat.title = f"{label}_2x2", f"{label}, 2x2"
        panels += [result, repeat]
        print(f"{job['name']} {label}  {metrics}", flush=True)

    for k, panel in enumerate(panels, start=1):
        panel.name = f"{k}_{panel.name}"
        panel.image.save(folder / f"{panel.name}.png")
    record["steps"] = {p.name: p.title + (f" ({p.tag})" if p.tag else "") for p in panels}
    sheet = build_sheet(panels, hole, header(record, config), generated_offset=half)
    sheet.save(folder / "sheet.png", compress_level=3)
    thumb = sheet.copy()
    thumb.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    thumb.save(folder / "thumb.jpg", quality=85)
    (folder / "run.json").write_text(json.dumps(record, indent=2))  # last: its presence means "done"


def tagline(m: dict) -> str:
    parts = [f"seam {m['seam_after']:.2f}"]
    if m.get("wrap_after") is not None:
        parts.append(f"wrap {m['wrap_after']:.2f}")
    if "kept_psnr" in m:
        parts.append(f"kept {m['kept_psnr']:.1f} dB")
    return "  ".join(parts)


COLUMNS = (
    "Generated once per column from the same noise; only the attention differs: band 0 is stock attention, "
    "band b sees b rows and columns of copies past each edge, rope is the nearest-copy RoPE of experiments 1-3."
)


def header(record: dict, config: dict) -> list[str]:
    """The lines on top of a sheet: enough to know what was run without opening run.json."""
    if record["dry_run"]:
        settings = "DRY RUN: inputs only, nothing was generated"
    else:
        settings = (
            f"{config['model_name']}   {config['num_steps']} steps   guidance {config['guidance']:g}"
            f"   seed {config['seed']}   bands {config['bands']}"
            + ("   + nearest-copy RoPE as control" if config["compare_rope"] else "")
        )
    if "tile" in record:
        exp = EXPERIMENTS[config["task"]]
        lines = [
            f"Experiment 4: copy tokens, inpaint   |   {record['name']}",
            f"Task of {exp.title.lower()}: {exp.about} {COLUMNS}",
            f"hole: {record['hole']} px across each seam   |   {settings}",
        ]
        prompt = config["prompt"]
    else:
        lines = [
            f"Experiment 4: copy tokens, text to image   |   {record['name']}",
            f"The picture from the prompt alone, {config['size']} x {config['size']}. {COLUMNS}",
            settings,
        ]
        prompt = record["prompt"]
    lines += [f"system: {config['system_prompt'] or '(none: bare user turn)'}", f"prompt: {prompt}"]
    if record["variants"]:
        first = next(iter(record["variants"].values()))["metrics"]
        seam = "   ".join(f"{label} {v['metrics']['seam_after']:.2f}" for label, v in record["variants"].items())
        before = f"before {first['seam_before']:.2f}, " if "seam_before" in first else ""
        lines.append(f"seam jump (1 = like any other line): {before}after:   {seam}")
    return lines


def write_report(run_dir: str) -> Path:
    """index.html and summary.csv from the run.json of every job; can be rerun at any time."""
    run = Path(run_dir)
    config = json.loads((run / "config.json").read_text())
    records = {r["name"]: r for r in (json.loads(p.read_text()) for p in run.glob("*/run.json"))}
    labels = [label for label, _ in variants(config)]
    columns = ["seam_after"] if config["task"] == "t2i" else ["seam_after", "wrap_after", "kept_psnr"]

    with (run / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["job", "variant", "band", "seam_before", *columns, "seconds", "sheet"])
        for r in (records[job["name"]] for job in config["jobs"] if job["name"] in records):
            for label, v in r["variants"].items():
                m = v["metrics"]
                writer.writerow([r["name"], label, v["band"], m.get("seam_before"), *[m[c] for c in columns], v["seconds"], f"{r['name']}/sheet.png"])

    out = [f"<!doctype html><meta charset=utf-8><title>{html.escape(run.name)}</title><style>{STYLE}</style>"]
    out.append(f"<h1>{html.escape(run.name)}: experiment 4, copy tokens, {'text to image' if config['task'] == 't2i' else 'inpaint'}</h1>")
    if config["dry_run"]:
        out.append("<p>Dry run: the model's inputs only, nothing was generated.</p>")
    else:
        out.append(
            f"<p>{html.escape(config['model_name'])}{' (TOY WEIGHTS: noise)' if config['toy'] else ''} &middot; "
            f"{config['num_steps']} steps &middot; guidance {config['guidance']:g} &middot; seed {config['seed']}"
            f" &middot; bands <code>{config['bands']}</code> tokens"
            f"{' &middot; nearest-copy RoPE as control' if config['compare_rope'] else ''}</p>"
        )
    task = (
        f"{config['size']} x {config['size']} from the prompt alone"
        if config["task"] == "t2i"
        else f"task <code>{config['task']}</code>, hole {config['hole']} px across each seam"
    )
    out.append(
        f"<p>{task} &middot; {len(config['jobs'])} jobs &middot; commit {html.escape(str(config['commit']))}</p>"
        f"<p>system: {html.escape(config['system_prompt'] or '(none: bare user turn)')}</p>"
        + (f"<p>prompt: {html.escape(config['prompt'])}</p>" if config["task"] != "t2i" else "")
    )

    measured = [r for r in records.values() if r["variants"]]
    if measured:
        out.append("<h2>Median over jobs</h2><table><tr><th>variant</th><th>jobs</th>")
        out += [f"<th>{c}</th>" for c in columns] + ["<th>seconds</th></tr>"]
        for label in labels:
            rows = [r["variants"][label] for r in measured if label in r["variants"]]
            out.append(f"<tr><td>{label}</td><td class=num>{len(rows)}</td>")
            for c in columns:
                values = [v["metrics"][c] for v in rows if v["metrics"][c] is not None]
                out.append(f"<td class=num>{fmt(statistics.median(values)) if values else ''}</td>")
            out.append(f"<td class=num>{fmt(statistics.median([v['seconds'] for v in rows]), 1) if rows else ''}</td></tr>")
        out.append("</table>")
        befores = [m["seam_before"] for r in measured for m in [next(iter(r["variants"].values()))["metrics"]] if "seam_before" in m]
        if befores:
            out.append(f"<p>seam jump before, median: {fmt(statistics.median(befores))}</p>")
        out.append("<dl>" + "".join(f"<dt><code>{c}</code></dt><dd>{html.escape(METRICS[c])}</dd>" for c in columns) + "</dl>")

    out.append("<h2>Sheets</h2><p>Click one for the full-resolution sheet; the folder link has every panel as its own PNG.</p>")
    out.append(f"<table class=grid><tr><th>job</th><th>sheet</th><th>variant: {' &middot; '.join(columns)}</th></tr>")
    for job in config["jobs"]:
        r, name = records.get(job["name"]), job["name"]
        out.append(f"<td class=name>{html.escape(name)}" + (f"<br><span class=m>{html.escape(job['prompt'])}</span>" if "prompt" in job else "") + "</td>")
        if r is None:
            out.append("<td class=m colspan=2>missing: see logs/</td></tr>")
            continue
        out.append(f"<td><a href='{name}/sheet.png'><img loading=lazy src='{name}/thumb.jpg'></a><a href='{name}/'>folder</a></td>")
        lines = [f"{label}: " + " &middot; ".join(fmt(v["metrics"][c], 1 if c == "kept_psnr" else 2) for c in columns) for label, v in r["variants"].items()]
        out.append(f"<td class=m>{'<br>'.join(lines)}</td></tr>")
    out.append("</table>")
    (run / "index.html").write_text("\n".join(out))
    return run / "index.html"


def main(run_dir: str, shard: int = 0, num_shards: int = 1):
    """One process, one GPU: the sweep on jobs[shard::num_shards]. config.json in the run folder says the rest."""
    config = json.loads((Path(run_dir) / "config.json").read_text())
    jobs = config["jobs"][shard::num_shards]
    print(f"shard {shard} of {num_shards}: {len(jobs)} jobs", flush=True)

    klein = None
    if not config["dry_run"]:
        settings = {k: config[k] for k in ("num_steps", "guidance")}
        if config["toy"]:
            klein = KleinCopy.toy(config["prompts"], config["system_prompt"], **settings)
        else:
            klein = KleinCopy.load(config["model_name"], config["prompts"], config["system_prompt"], **settings)

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
