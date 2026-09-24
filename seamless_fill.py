#!/usr/bin/env python3
"""
Make a four-way seamless floral/texture tile with FLUX.1 Fill.

Pipeline
--------
1. Fix left/right seam:
   horizontal roll -> vertical inpaint band -> roll back
2. Fix top/bottom seam:
   vertical roll -> horizontal inpaint band -> roll back
3. Fix the remaining corner interaction:
   2D roll -> center square inpaint -> roll back

Every intermediate image is saved in a per-input debug folder.

Example
-------
python seamless_fill.py small_test_roses.png \
    --size 512 \
    --lr-mask 128 \
    --ud-mask 128 \
    --center-mask 128

You can also resize non-square:
python seamless_fill.py input.png --width 768 --height 512

Outputs are written to:
    outputs/<input_stem>_<timestamp>/
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import FluxFillPipeline


DEFAULT_PROMPT = (
    "continue the surrounding image content naturally, "
    "preserving the same visual style, colors, texture, structure, and level of detail, "
    "extend nearby shapes and patterns smoothly, "
    "and keep the result consistent with the surrounding image"
)

CENTER_PROMPT = (
    "continue the surrounding image content naturally from all sides, "
    "preserving the same visual style, colors, texture, structure, and detail level, "
    "make only a small natural continuation of nearby content, "
    "do not create any new central object, block, label, frame, or text, "
    "and keep the result visually consistent with the surrounding image"
)


def log(msg: str) -> None:
    print(f"[seamless-fill] {msg}", flush=True)


def save_mask(mask_np: np.ndarray, path: Path) -> Image.Image:
    mask = Image.fromarray(mask_np.astype(np.uint8), mode="L")
    mask.save(path)
    return mask


def make_preview(tile: Image.Image, nx: int, ny: int) -> Image.Image:
    w, h = tile.size
    out = Image.new("RGB", (w * nx, h * ny))
    for iy in range(ny):
        for ix in range(nx):
            out.paste(tile, (ix * w, iy * h))
    return out


def validate_size(w: int, h: int) -> None:
    # FLUX/VAE pipelines are happiest with dimensions divisible by 16.
    if w % 16 != 0 or h % 16 != 0:
        raise ValueError(
            f"Requested size {w}x{h} is not divisible by 16. "
            "Please use dimensions such as 512, 768, 1024, etc."
        )


def resize_image(img: Image.Image, width: int | None, height: int | None, size: int | None) -> Image.Image:
    if size is not None:
        width = height = size

    if width is None and height is None:
        return img

    if width is None or height is None:
        raise ValueError("Use --size N, or provide both --width W and --height H.")

    validate_size(width, height)

    if img.size == (width, height):
        return img

    log(f"Resizing input from {img.size[0]}x{img.size[1]} to {width}x{height}")
    return img.resize((width, height), Image.Resampling.LANCZOS)


def run_fill(
    pipe: FluxFillPipeline,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
) -> Image.Image:
    w, h = image.size

    return pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=h,
        width=w,
        guidance_scale=guidance,
        num_inference_steps=steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0].convert("RGB")


def fix_left_right(
    pipe: FluxFillPipeline,
    tile: Image.Image,
    outdir: Path,
    mask_width: int,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
) -> Image.Image:
    w, h = tile.size
    if not 0 < mask_width < w:
        raise ValueError(f"--lr-mask must be between 1 and {w - 1}")

    log("Step 1/3: fixing left/right seam")

    arr = np.array(tile)
    rolled_arr = np.roll(arr, shift=w // 2, axis=1)
    rolled = Image.fromarray(rolled_arr)
    rolled.save(outdir / "01_lr_rolled_before.png")

    cx = w // 2
    x0 = cx - mask_width // 2
    x1 = x0 + mask_width

    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[:, x0:x1] = 255
    mask = save_mask(mask_np, outdir / "01_lr_mask.png")

    filled = run_fill(
        pipe=pipe,
        image=rolled,
        mask=mask,
        prompt=prompt,
        steps=steps,
        guidance=guidance,
        seed=seed,
    )
    filled.save(outdir / "01_lr_rolled_after.png")

    result_arr = np.roll(np.array(filled), shift=-(w // 2), axis=1)
    result = Image.fromarray(result_arr)
    result.save(outdir / "01_lr_result.png")

    make_preview(result, 2, 1).save(outdir / "01_lr_preview_2x1.png")
    log("Step 1/3 complete")

    return result


def fix_up_down(
    pipe: FluxFillPipeline,
    tile: Image.Image,
    outdir: Path,
    mask_height: int,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
) -> Image.Image:
    w, h = tile.size
    if not 0 < mask_height < h:
        raise ValueError(f"--ud-mask must be between 1 and {h - 1}")

    log("Step 2/3: fixing top/bottom seam")

    arr = np.array(tile)
    rolled_arr = np.roll(arr, shift=h // 2, axis=0)
    rolled = Image.fromarray(rolled_arr)
    rolled.save(outdir / "02_ud_rolled_before.png")

    cy = h // 2
    y0 = cy - mask_height // 2
    y1 = y0 + mask_height

    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[y0:y1, :] = 255
    mask = save_mask(mask_np, outdir / "02_ud_mask.png")

    filled = run_fill(
        pipe=pipe,
        image=rolled,
        mask=mask,
        prompt=prompt,
        steps=steps,
        guidance=guidance,
        seed=seed + 1,
    )
    filled.save(outdir / "02_ud_rolled_after.png")

    result_arr = np.roll(np.array(filled), shift=-(h // 2), axis=0)
    result = Image.fromarray(result_arr)
    result.save(outdir / "02_ud_result.png")

    make_preview(result, 1, 2).save(outdir / "02_ud_preview_1x2.png")
    make_preview(result, 2, 2).save(outdir / "02_ud_preview_2x2.png")
    log("Step 2/3 complete")

    return result


def fix_center_coupling(
    pipe: FluxFillPipeline,
    tile: Image.Image,
    outdir: Path,
    center_mask: int,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
) -> Image.Image:
    w, h = tile.size
    if not 0 < center_mask < min(w, h):
        raise ValueError(
            f"--center-mask must be between 1 and {min(w, h) - 1}"
        )

    log("Step 3/3: fixing remaining corner/cross coupling")

    arr = np.array(tile)
    rolled_arr = np.roll(arr, shift=h // 2, axis=0)
    rolled_arr = np.roll(rolled_arr, shift=w // 2, axis=1)
    rolled = Image.fromarray(rolled_arr)
    rolled.save(outdir / "03_center_rolled_before.png")

    cx, cy = w // 2, h // 2
    x0 = cx - center_mask // 2
    x1 = x0 + center_mask
    y0 = cy - center_mask // 2
    y1 = y0 + center_mask

    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[y0:y1, x0:x1] = 255
    mask = save_mask(mask_np, outdir / "03_center_mask.png")

    filled = run_fill(
        pipe=pipe,
        image=rolled,
        mask=mask,
        prompt=prompt,
        steps=steps,
        guidance=guidance,
        seed=seed + 2,
    )
    filled.save(outdir / "03_center_rolled_after.png")

    result_arr = np.roll(np.array(filled), shift=-(h // 2), axis=0)
    result_arr = np.roll(result_arr, shift=-(w // 2), axis=1)
    result = Image.fromarray(result_arr)
    result.save(outdir / "03_center_result.png")

    log("Step 3/3 complete")
    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Turn an image into a four-way seamless tile using FLUX.1 Fill."
    )

    p.add_argument("input", type=Path, help="Input image path")
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for debug/output folders (default: outputs)",
    )

    # Resize options
    resize = p.add_argument_group("resize")
    resize.add_argument(
        "--size",
        type=int,
        default=None,
        help="Resize to NxN before processing, e.g. --size 512",
    )
    resize.add_argument("--width", type=int, default=None, help="Resize width")
    resize.add_argument("--height", type=int, default=None, help="Resize height")

    # Mask sizes
    masks = p.add_argument_group("mask sizes")
    masks.add_argument(
        "--lr-mask",
        type=int,
        default=128,
        help="Width of vertical inpaint band for left/right seam (default: 128)",
    )
    masks.add_argument(
        "--ud-mask",
        type=int,
        default=128,
        help="Height of horizontal inpaint band for top/bottom seam (default: 128)",
    )
    masks.add_argument(
        "--center-mask",
        type=int,
        default=128,
        help="Side length of final center square mask (default: 128)",
    )

    # Sampling
    sample = p.add_argument_group("sampling")
    sample.add_argument("--steps", type=int, default=50, help="Steps for stages 1/2")
    sample.add_argument(
        "--guidance", type=float, default=30.0, help="Guidance for stages 1/2"
    )
    sample.add_argument(
        "--center-steps", type=int, default=30, help="Steps for center stage"
    )
    sample.add_argument(
        "--center-guidance",
        type=float,
        default=10.0,
        help="Guidance for center stage",
    )
    sample.add_argument("--seed", type=int, default=0)

    # Prompt/model
    p.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt for left/right and top/bottom repair",
    )
    p.add_argument(
        "--center-prompt",
        type=str,
        default=CENTER_PROMPT,
        help="Prompt for final center repair",
    )
    p.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-Fill-dev",
        help="Diffusers model id/path",
    )

    p.add_argument(
        "--preview-grid",
        type=int,
        default=4,
        help="Final NxN tiled preview size (default: 4)",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.size is not None and (args.width is not None or args.height is not None):
        raise ValueError("Use either --size or --width/--height, not both.")

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{input_path.stem}_{timestamp}"
    outdir = args.output_root / run_name
    outdir.mkdir(parents=True, exist_ok=False)

    log(f"Input: {input_path}")
    log(f"Output folder: {outdir}")

    img = Image.open(input_path).convert("RGB")
    img = resize_image(img, args.width, args.height, args.size)

    w, h = img.size
    validate_size(w, h)

    img.save(outdir / "00_input.png")
    log(f"Working size: {w}x{h}")

    log(f"Loading model: {args.model}")
    pipe = FluxFillPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    log("Model loaded")

    stage1 = fix_left_right(
        pipe=pipe,
        tile=img,
        outdir=outdir,
        mask_width=args.lr_mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )

    stage2 = fix_up_down(
        pipe=pipe,
        tile=stage1,
        outdir=outdir,
        mask_height=args.ud_mask,
        prompt=args.prompt,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )

    final = fix_center_coupling(
        pipe=pipe,
        tile=stage2,
        outdir=outdir,
        center_mask=args.center_mask,
        prompt=args.center_prompt,
        steps=args.center_steps,
        guidance=args.center_guidance,
        seed=args.seed,
    )

    final_path = outdir / "final_seamless_tile.png"
    final.save(final_path)

    n = args.preview_grid
    if n > 0:
        preview = make_preview(final, n, n)
        preview.save(outdir / f"final_preview_{n}x{n}.png")

    log("Pipeline complete")
    log(f"Final tile: {final_path}")
    if n > 0:
        log(f"Preview: {outdir / f'final_preview_{n}x{n}.png'}")


if __name__ == "__main__":
    main()
