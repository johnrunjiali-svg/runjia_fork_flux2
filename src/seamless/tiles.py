"""A picture as a tile: cut it somewhere else, paint part of it white, repeat it, measure its seam.

Numpy and PIL only, so everything here runs on a laptop. The one idea worth stating is `roll`. Copy a
tile 2x2 and slide a 1x1 window over the copies: whatever the window holds is the same tile cut at a
different place, and the cut the original had -- its left/right edge, its top/bottom edge -- is now
a line *inside* the picture. Rolled by half, both of those lines cross in the middle.

A line is named by the pixel on its right (or below it): the vertical line x runs between columns
x - 1 and x, so line 0 is the tile's own edge, where column w - 1 meets column 0 of the next copy.
"""

import numpy as np
from PIL import Image

WHITE = (255, 255, 255)


def roll(image: Image.Image, dx: int = 0, dy: int = 0) -> Image.Image:
    """The tile cut at a different place: the 1x1 window whose top-left corner sits at (dx, dy) on a
    2x2 board of copies. The tile's own edges end up on the lines x = w - dx and y = h - dy."""
    return Image.fromarray(np.roll(np.asarray(image), shift=(-dy, -dx), axis=(0, 1)))


def unroll(image: Image.Image, dx: int = 0, dy: int = 0) -> Image.Image:
    """Undo `roll`, so a result comes back in the framing of the tile it was made from."""
    return roll(image, -dx, -dy)


def tiled(image: Image.Image, copies: int = 2) -> Image.Image:
    """copies x copies of the tile edge to edge: what a repeat of it actually looks like."""
    return Image.fromarray(np.tile(np.asarray(image), (copies, copies, 1)))


def seam_band(size: tuple[int, int], x: int, y: int, thickness: int) -> np.ndarray:
    """[h, w] bool: every pixel within thickness / 2 of the vertical line x or the horizontal line y,
    wrapping around the edges. At (w // 2, h // 2) it is a cross through the middle; at (0, 0) the
    same cross sits on the edges and reads as a frame thickness / 2 wide. They are one mask, rolled."""
    w, h = size
    assert thickness % 2 == 0 and 0 < thickness < min(w, h), f"thickness {thickness} for a {w}x{h} tile"
    cols = (np.arange(w) - x + thickness // 2) % w < thickness
    rows = (np.arange(h) - y + thickness // 2) % h < thickness
    return rows[:, None] | cols[None, :]


def paint(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = WHITE) -> Image.Image:
    """The picture with everything under `mask` set to one flat colour. White is the value an
    image-to-image model reads as "this is missing" rather than as content to keep."""
    pixels = np.array(image.convert("RGB"))
    assert pixels.shape[:2] == mask.shape, f"mask {mask.shape} does not fit image {pixels.shape[:2]}"
    pixels[mask] = color
    return Image.fromarray(pixels)


def center_crop(image: Image.Image, border: int) -> Image.Image:
    """The picture without its outer `border` pixels: a tile that no longer repeats."""
    return image.crop((border, border, image.width - border, image.height - border))


def pad(image: Image.Image, border: int, color: tuple[int, int, int] = WHITE) -> Image.Image:
    """The picture centred on a canvas `border` pixels larger on every side."""
    canvas = Image.new("RGB", (image.width + 2 * border, image.height + 2 * border), color)
    canvas.paste(image, (border, border))
    return canvas


def resize_periodic(image: Image.Image, size: int) -> Image.Image:
    """A square tile resampled to size x size as one period of an endless repeat.

    A plain resize treats the edges as the end of the picture: the filter runs out of pixels there
    and the two sides of the seam are resampled without seeing each other. Here the tile is first
    continued past every edge with its own wrapped copies, so the seam is filtered like any other
    line of the picture -- no better and no worse than it was."""
    image = image.convert("RGB")
    assert image.width == image.height, f"{image.size}: a crop would cut the repeat, a squash distort it"
    n = image.width
    margin = min(n, 4 * (n // size + 2))  # past the reach of the Lanczos kernel (3 px at the coarser scale)
    wrapped = np.pad(np.asarray(image), ((margin, margin), (margin, margin), (0, 0)), mode="wrap")
    box = (margin, margin, margin + n, margin + n)  # resize filters with what lies outside its box
    return Image.fromarray(wrapped).resize((size, size), Image.Resampling.LANCZOS, box=box)


def seam_jump(image: Image.Image, x: int = 0, y: int = 0) -> float:
    """How much the vertical line x and the horizontal line y stand out as cuts: 1 is "like any other
    line of this picture". The defaults measure the tile's own edges, i.e. the seam of its repeat.

    For every line of the picture, take the mean absolute difference between the pixels on its two
    sides (wrapping, so the edge counts as a line). A clean line scores what its neighbours score; a
    cut, where the two sides were never drawn together, scores higher. The value returned is the
    line's score over the median score of all parallel lines, the worse of the two directions.

    It only sees an abrupt change across one pixel. A join that is smooth but wrong -- a stem that
    bends, a motif that changes colour over a few pixels -- needs the eye, which is what the sheet
    is for."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    ratios = []
    for axis, line in ((1, x), (0, y)):
        jumps = np.abs(pixels - np.roll(pixels, 1, axis=axis)).mean(axis=(1 - axis, 2))
        ratios.append(float(jumps[line % len(jumps)] / max(float(np.median(jumps)), 1e-6)))
    return max(ratios)


def psnr(a: Image.Image, b: Image.Image, where: np.ndarray | None = None) -> float:
    """Peak signal to noise ratio in dB over the pixels selected by `where` (all of them by default)."""
    diff = np.asarray(a.convert("RGB"), dtype=np.float32) - np.asarray(b.convert("RGB"), dtype=np.float32)
    mse = float(np.square(diff if where is None else diff[where]).mean())
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))
