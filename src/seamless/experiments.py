"""The three experiments, as the pictures each one makes on the way from a tile to a result.

Every experiment ends in the same act: a 1024 x 1024 picture with a white hole is handed to FLUX.2
klein as a reference image, with the instruction to fill the white, and the output is generated on
the torus (flux2/torus.py: the seamless RoPE, no extra prompt, no extra guidance branch). The
instruction has two parts, a system turn that tells the model it is an inpainter and the user
turn that says what to fill; see SYSTEM_PROMPT for why the first is not optional. They
differ in where the picture comes from and in how it is framed when the model sees it.

  1_seam_fix        An almost seamless tile. Roll it by half so its seams cross in the middle, paint
                    a cross of `band` px over them, generate, roll back.
  2_outpaint_frame  A tile that does not repeat: the centre crop of the same tile, `border` px
                    removed on every side. Put it back on a white canvas of the full size, so the
                    hole is a frame around it, and generate.
  3_outpaint_cross  The same crop on the same canvas, rolled by half before the model sees it: the
                    frame becomes a cross in the middle. Generate, roll back.

Two of the three are the same run. A frame `border` wide around the tile and a cross 2 x `border`
wide through the rolled tile are one mask (tiles.seam_band), and painting it erases the same pixels
whichever way it is described, so with band = 2 x border experiments 1 and 3 hand the model a
byte-identical picture. `run_tile` generates once and reports it twice: once against the original
tile (was the seam repaired, and what did it cost?), once against the crop (did a tile that did not
repeat become one that does?). Experiment 2 is the other run, and its input is that same picture
before the roll -- so 2 against 3 isolates the one thing that changed, where the hole sits.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .sheet import Panel, build_sheet
from .tiles import center_crop, pad, paint, psnr, roll, seam_band, seam_jump, tiled, unroll

# The text encoder's system turn, in front of the fill prompt. The klein models were conditioned on a
# bare user turn, and with only that the instruction does not reliably take: some tiles come back
# with the white cross untouched, or with one white band left across the middle. Telling the model
# what kind of model it is fixes it (found with the "inpainter" preset of the dev branch's ref_web).
SYSTEM_PROMPT = (
    "You are an image completion model. Blank white regions of the input are holes to be filled. "
    "You reconstruct what the surrounding image implies should be there, matching its style, "
    "colour and scale, and you reproduce the rest of the image unchanged."
)

FILL_PROMPT = (
    "Fill in the blank white regions of this image. "
    "Continue the surrounding pattern straight across them so that nothing marks where they were: "
    "the same style, colours, texture, line weight, density and scale of motif, "
    "and shapes that run into the white area carried on to meet what is on the other side. "
    "Everything outside the white regions stays exactly as it is. "
    "Do not add any new object, frame, border, label or text."
)


def rope_label(config: dict) -> str:
    """How the RoPE was made periodic, for a header: 'nearest', 'quantized', 'quantized r r f f k k ...' or 'OFF'."""
    if not config["wrap"]:
        return "OFF"
    label = config.get("rope", "nearest")
    if config.get("rules"):
        label += " " + " ".join(str(r) if isinstance(r, int) else r[0] for r in config["rules"])
    return label


@dataclass(frozen=True)
class Experiment:
    name: str  # the folder it writes to
    title: str
    about: str
    crop: bool  # start from the centre crop, a tile that does not repeat, instead of the tile itself
    rolled: bool  # show the model the picture rolled by half: the hole is a cross, not a frame


EXPERIMENTS = {
    e.name: e
    for e in (
        Experiment(
            "1_seam_fix",
            "Experiment 1: seam fix",
            "An almost seamless tile is rolled by half, a white cross is painted over its seams, "
            "and the model fills the cross.",
            crop=False,
            rolled=True,
        ),
        Experiment(
            "2_outpaint_frame",
            "Experiment 2: outpaint, hole as a frame",
            "The centre crop of the tile (it does not repeat) is put back on a white canvas, "
            "and the model fills the frame around it.",
            crop=True,
            rolled=False,
        ),
        Experiment(
            "3_outpaint_cross",
            "Experiment 3: outpaint, hole as a cross",
            "The same crop on the same white canvas, rolled by half so the frame becomes a cross "
            "in the middle, and the model fills the cross.",
            crop=True,
            rolled=True,
        ),
    )
}


def before(exp: Experiment, tile: Image.Image, hole: int) -> tuple[list[Panel], tuple[int, int]]:
    """Everything up to the model's input, which is the last panel. `hole` is how thick the white
    region is across a seam: the width of the cross, twice the width of the frame. Also returns the
    offset the input is rolled by."""
    half = (tile.width // 2, tile.height // 2) if exp.rolled else (0, 0)
    panels = [
        Panel("original", "original tile", tile, marks="hole"),
        Panel("original_2x2", "original, 2x2", tiled(tile), copies=2, marks="seam hole"),
    ]
    if exp.crop:
        crop = center_crop(tile, hole // 2)
        panels += [
            Panel("crop", f"centre {crop.width} x {crop.height} crop", crop),
            Panel("crop_2x2", "crop, 2x2", tiled(crop), copies=2, marks="seam"),
            Panel("padded", f"crop + {hole // 2} px white frame", pad(crop, hole // 2), marks="hole"),
        ]
        if exp.rolled:
            rolled = roll(panels[-1].image, *half)
            panels.append(Panel("rolled", "rolled by 1/2", rolled, offset=half, marks="seam hole"))
    else:
        rolled = roll(tile, *half)
        masked = paint(rolled, seam_band(tile.size, *half, hole))
        panels += [
            Panel("rolled", "rolled by 1/2", rolled, offset=half, marks="seam"),
            Panel("masked", f"{hole} px cross painted white", masked, offset=half, marks="seam hole"),
        ]
    panels[-1].tag = "model input"
    return panels, half


def after(exp: Experiment, generated: Image.Image, half: tuple[int, int]) -> list[Panel]:
    """From the model's output to the tile, which is the second to last panel, and its 2x2 repeat."""
    panels = [Panel("generated", "generated", generated, offset=half, marks="seam hole")]
    if exp.rolled:
        panels.append(
            Panel("unrolled", "generated, rolled back", unroll(generated, *half), marks="wrap hole")
        )
    result = panels[-1]
    result.tag = "result"
    panels.append(
        Panel(f"{result.name}_2x2", "result, 2x2", tiled(result.image), copies=2, marks="seam wrap hole")
    )
    return panels


def measure(exp: Experiment, panels: list[Panel], hole: int, half: tuple[int, int]) -> dict:
    """Four numbers for the summary table; tiles.seam_jump says what a jump is and what it misses."""
    by_name = {p.name: p.image for p in panels}
    reference = next(p.image for p in panels if p.tag == "model input")
    result = next(p.image for p in panels if p.tag == "result")
    kept = ~seam_band(reference.size, *half, hole)  # `half` is where the seam sits in the model's framing
    return {
        # the join of the tile the experiment starts from, and of the tile it ends with
        "seam_before": round(seam_jump(by_name["crop" if exp.crop else "original"]), 2),
        "seam_after": round(seam_jump(result), 2),
        # where the generated picture's own edges meet once it is rolled back: the model was asked
        # to leave them alone, so this is 1 unless it redrew what it should have kept
        "wrap_after": round(seam_jump(result, *half), 2) if exp.rolled else None,
        # how faithfully the model reproduced what it was told to keep
        "kept_psnr": round(psnr(reference, by_name["generated"], kept), 2),
    }


def run_tile(
    tile_path: str | Path,
    run_dir: str | Path,
    config: dict,
    generate: Callable[[Image.Image, str], Image.Image] | None,
):
    """Every experiment of the run on one tile. Writes <run_dir>/<experiment>/<tile>/ with one PNG
    per step, sheet.png, thumb.jpg and run.json. `generate(reference, prompt)` is the model;
    None is a dry run that stops at the model's input, to look at the masks before spending GPU time."""
    tile_path, run_dir = Path(tile_path), Path(run_dir)
    tile = Image.open(tile_path).convert("RGB")
    generated_from: dict[
        str, tuple[Image.Image, str, float]
    ] = {}  # model input -> output, experiment, seconds

    for name in config["experiments"]:
        exp = EXPERIMENTS[name]
        hole = 2 * config["border"] if exp.crop else config["band"]
        folder = run_dir / exp.name / tile_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        panels, half = before(exp, tile, hole)
        record = {
            "experiment": exp.name,
            "tile": tile_path.stem,
            "tile_path": str(tile_path),
            "hole": hole,
            "roll": list(half),
            "dry_run": generate is None,
        }

        if generate is not None:
            reference = panels[-1].image
            key = hashlib.sha256(reference.tobytes()).hexdigest()
            if key not in generated_from:
                started = time.time()
                generated_from[key] = (generate(reference, config["prompt"]), exp.name, time.time() - started)
            generated, source, seconds = generated_from[key]
            panels += after(exp, generated, half)
            record |= {
                "seconds": round(seconds, 2),
                # set when another experiment handed the model the very same picture: one generation, two reports
                "same_generation_as": source if source != exp.name else None,
                "metrics": measure(exp, panels, hole, half),
            }

        for k, panel in enumerate(panels, start=1):
            panel.name = f"{k}_{panel.name}"
            panel.image.save(folder / f"{panel.name}.png")
        record["steps"] = {p.name: p.title + (f" ({p.tag})" if p.tag else "") for p in panels}
        sheet = build_sheet(panels, hole, header(exp, record, config), generated_offset=half)
        sheet.save(folder / "sheet.png", compress_level=3)
        thumb = sheet.copy()
        thumb.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        thumb.save(folder / "thumb.jpg", quality=85)
        (folder / "run.json").write_text(json.dumps(record, indent=2))  # last: its presence means "done"
        print(
            f"{exp.name}/{tile_path.stem}" + (f"  {record['metrics']}" if "metrics" in record else ""),
            flush=True,
        )


def header(exp: Experiment, record: dict, config: dict) -> list[str]:
    """The lines on top of a sheet: enough to know what was run without opening run.json."""
    if record["dry_run"]:
        settings = "DRY RUN: inputs only, nothing was generated"
    else:
        settings = (
            f"{config['model_name']}   {config['num_steps']} steps   guidance {config['guidance']:g}"
            f"   seamless RoPE {rope_label(config)}   seed {config['seed']}"
            f"   {record['seconds']:.1f} s"
        )
        if record["same_generation_as"]:
            settings += (
                f"   (same model input as {record['same_generation_as']}: one generation, shown twice)"
            )
    lines = [
        f"{exp.title}   |   {record['tile']}",
        exp.about,
        f"hole: {record['hole']} px across each seam   |   {settings}",
        f"system: {config['system_prompt'] or '(none: bare user turn)'}",
        f"prompt: {config['prompt']}",
    ]
    if "metrics" in record:
        m = record["metrics"]
        wrap = "" if m["wrap_after"] is None else f"   wrap jump {m['wrap_after']:.2f}"
        lines.append(
            f"seam jump {m['seam_before']:.2f} -> {m['seam_after']:.2f}{wrap}   (1 = like any other line)"
            f"   |   kept region vs input: {m['kept_psnr']:.1f} dB"
        )
    return lines
