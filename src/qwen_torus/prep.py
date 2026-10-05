"""Pictures in and out of a run: rolling a tile, the prompts, the views the web page shows.

Pure PIL / numpy, no model. A copy of the few helpers of src/flux2/masked_ref.py this package needs,
kept here so that qwen_torus imports nothing from flux2.

The two ways of telling Qwen-Image 2.1 what to change in a reference, both just pictures:

- Paint a region flat white and ask for it to be filled (FILL_PROMPT). The hole is visible content
  the instruction can point at, and the rest of the picture is the style to match.
- Draw a coloured outline around a region and name the colour in the prompt (OUTLINE_PROMPT). The
  model edits what is inside and is told to drop the line. Several colours, several instructions.

Rolling is the seam trick: a tile copied 2x2 and cut elsewhere is the same tile with its own edges
now running through the middle, where they can be painted over and redrawn.
"""

import numpy as np
from PIL import Image

# Starting points, deliberately short. Tune them in the web page.
FILL_PROMPT = "Fill the white areas so they continue the surrounding pattern seamlessly. Keep everything else unchanged."

OUTLINE_PROMPT = (
    "Inside the {color} outline, {edit}. Remove the outline itself and keep everything else unchanged."
)

TORUS_PROMPT = (
    "A single seamless repeating tile: the left edge continues into the right edge and the top into "
    "the bottom, with no border and no visible repetition inside."
)

OUTLINE_COLORS = {
    "red": (230, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 80, 230),
    "yellow": (240, 210, 20),
    "magenta": (220, 40, 200),
    "cyan": (20, 190, 210),
}


def roll(image: Image.Image, dx: int = 0, dy: int = 0) -> Image.Image:
    """The tile cut at a different place: the 1x1 window whose top-left corner sits at (dx, dy) on a
    2x2 board of copies. Equivalent to np.roll by (-dy, -dx); `unroll` puts it back."""
    return Image.fromarray(np.roll(np.asarray(image), shift=(-dy, -dx), axis=(0, 1)))


def unroll(image: Image.Image, dx: int = 0, dy: int = 0) -> Image.Image:
    return roll(image, -dx, -dy)


def seam(size: tuple[int, int], dx: int = 0, dy: int = 0) -> tuple[int, int]:
    """Where the tile's own edges ended up after `roll(image, dx, dy)`: the column and the row."""
    w, h = size
    return (w - dx) % w, (h - dy) % h


def band(size: tuple[int, int], axis: str, center: int, thickness: int) -> np.ndarray:
    """[h, w] bool stripe `thickness` wide centred on `center`, wrapping around the edges."""
    assert axis in ("v", "h"), axis
    w, h = size
    n = w if axis == "v" else h
    line = (np.arange(n) - (center - thickness // 2)) % n < min(thickness, n)
    mask = np.zeros((h, w), dtype=bool)
    if axis == "v":
        mask[:, line] = True
    else:
        mask[line, :] = True
    return mask


def cross(size: tuple[int, int], dx: int, dy: int, thickness: int) -> np.ndarray:
    """Both seams of `roll(image, dx, dy)` at once: all four edges, corners included, in one pass."""
    x, y = seam(size, dx, dy)
    return band(size, "v", x, thickness) | band(size, "h", y, thickness)


def paint(image: Image.Image, mask: np.ndarray, color=(255, 255, 255)) -> Image.Image:
    pixels = np.array(image.convert("RGB"))
    assert pixels.shape[:2] == mask.shape, f"mask {mask.shape} does not fit image {pixels.shape[:2]}"
    pixels[mask] = color
    return Image.fromarray(pixels)


def prepare(
    image: Image.Image, dx=0, dy=0, masks=None, color=(255, 255, 255)
) -> tuple[Image.Image, np.ndarray]:
    """Roll, then paint: the reference the model is shown, and the mask that made it."""
    rolled = roll(image, dx, dy)
    masks = [] if masks is None else ([masks] if isinstance(masks, np.ndarray) else masks)
    mask = np.zeros((rolled.height, rolled.width), dtype=bool)
    for m in masks:
        mask |= m
    return paint(rolled, mask, color), mask


def views(tile: Image.Image, dx: int = 0, dy: int = 0) -> dict[str, Image.Image]:
    """The tile, 2x2 copies pulled apart, 2x2 copies joined, and (if it was rolled) the tile rolled
    back to the framing the studio started from."""
    w, h = tile.size
    gap = max(4, w // 48)
    grid = Image.new("RGB", (2 * w + gap, 2 * h + gap), "white")
    joined = Image.new("RGB", (2 * w, 2 * h))
    for top in (0, h):
        for left in (0, w):
            joined.paste(tile, (left, top))
            grid.paste(tile, (left + (left > 0) * gap, top + (top > 0) * gap))
    out = {"tile": tile, "grid": grid, "seamless": joined}
    if dx or dy:
        out["unrolled"] = unroll(tile, dx, dy)
    return out


def seam_ratio(field) -> tuple[float, float]:
    """How visible the wrap is, as two numbers (rows, columns): the mean absolute difference between
    the last and the first row (column) divided by the mean absolute difference between adjacent rows
    (columns) inside. About 1 for a field that really repeats, several for one with an edge. Takes a
    PIL image, a uint8 [H, W, 3] array/tensor, or any [H, W, C] float field such as a latent grid, so
    the same number can be read before the decoder (the transformer's seam) and after it (the pixels')."""
    x = np.asarray(field.detach().cpu() if hasattr(field, "detach") else field, dtype=np.float64)
    wrap_h = np.abs(x[0] - x[-1]).mean()
    wrap_w = np.abs(x[:, 0] - x[:, -1]).mean()
    inner_h = np.abs(x[1:] - x[:-1]).mean()
    inner_w = np.abs(x[:, 1:] - x[:, :-1]).mean()
    return float(wrap_h / max(inner_h, 1e-12)), float(wrap_w / max(inner_w, 1e-12))


def fit_area(image: Image.Image, max_pixels: int, multiple: int = 32) -> Image.Image:
    """Shrink (never enlarge) to at most `max_pixels`, keeping the aspect ratio, sides a multiple of
    `multiple` -- 32 for Qwen-Image 2.1, whose vision slots are 2x2 tokens of 16 px."""
    w, h = image.size
    k = min(1.0, (max_pixels / (w * h)) ** 0.5)
    w, h = max(multiple, int(w * k) // multiple * multiple), max(multiple, int(h * k) // multiple * multiple)
    return image if (w, h) == image.size else image.resize((w, h), Image.Resampling.LANCZOS)
