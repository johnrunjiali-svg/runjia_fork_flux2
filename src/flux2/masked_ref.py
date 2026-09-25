"""Roll a tile, paint the part to redraw pure white, and hand that picture to FLUX.2 as a reference.

The problem this replaces
-------------------------
seamless_fill.py fixes a tile's seams with FLUX.1 Fill: roll the seam into the middle, give the
model the picture plus a mask, and it inpaints the band. It works, but it is one model doing one
task. FLUX.2 image-to-image is stronger and far more flexible -- except that handing it a whole
picture and asking for an inpainting-shaped edit does nothing: it was tuned to *edit what it is
told to edit* ("remove the glasses", "add a hat"), and an instruction about a region it cannot see
reads as "leave it alone", so it copies its input through.

Painting the region out first turns that weakness into the interface. A white hole is visible
content: the model can see exactly where it is, the instruction "fill the white area" names
something in the picture, and the rest of the reference is the style, palette and scale it has to
match. No mask channel, no inpainting head, no init latent -- the run is plain text+image-to-image
and the sampler starts from pure noise, so `generate` gets init_image=None and t_start=1.

    tile = Image.open("roses.png")
    ref, mask = prepare(tile, dx=tile.width // 2, masks=[band(tile.size, "v", tile.width // 2, 128)])
    out = generate_from_reference(pipe, FILL_PROMPT, ref, seed=0)[0]     # uint8 [H, W, 3]

Rolling
-------
`roll` is the whole seam trick and it is worth stating once. Copy a tile 2x2 and slide a 1x1 window
over the copies: whatever the window holds is the same tile cut at a different place, and the cut
the original had -- its left/right edge, its top/bottom edge -- is now a line *inside* the picture.
Paint over that line, have the model draw across it, and the join it drew is the join the tile's
edges will make when it is repeated. `seam` says where those lines are for a given offset, which is
what the band presets centre themselves on.

None of this needs a torus: rolling and masking are just pictures. Seamlessness comes from `wrap`
inside `generate` (torus.py), and turning it off -- geo_guidance=0, wrap=(False, False) -- leaves
an ordinary FLUX.2 edit of a masked picture, which is useful on its own.
"""

import numpy as np
from PIL import Image
from torch import Tensor

from .torus import TORUS_PROMPT
from .torus_generate import TorusPipe, generate

WHITE = (255, 255, 255)

FILL_PROMPT = (
    "Fill in the blank white regions of this image. "
    "Continue the surrounding pattern straight across them so that nothing marks where they were: "
    "the same style, colours, texture, line weight, density and scale of motif, "
    "and shapes that run into the white area carried on to meet what is on the other side. "
    "Everything outside the white regions stays exactly as it is. "
    "Do not add any new object, frame, border, label or text."
)

# Nothing here is tuned; they are starting points to compare. The klein models saw a bare user turn
# in training, so "" (no system turn at all) is the honest baseline to measure the others against.
SYSTEM_PRESETS: dict[str, str] = {
    "none (as trained)": "",
    "image editor": (
        "You are an image editing model. You apply the requested edit to the given image and change "
        "nothing else: composition, colours, lighting and every region the instruction does not name "
        "stay exactly as they are."
    ),
    "pattern designer": (
        "You are a textile and surface pattern designer. You work in flat repeating motifs and keep "
        "one consistent style, palette, line weight and density of motif across the whole surface, "
        "with no focal point, no border and no visible seam."
    ),
    "inpainter": (
        "You are an image completion model. Blank white regions of the input are holes to be filled. "
        "You reconstruct what the surrounding image implies should be there, matching its style, "
        "colour and scale, and you reproduce the rest of the image unchanged."
    ),
}


def crop_to_tile(
    image: Image.Image, width: int | None = None, height: int | None = None, multiple: int = 16
) -> Image.Image:
    """The picture as a tile the pipeline can work on: RGB, sides a multiple of 16 (the token size).

    With a width and height it is a cover crop -- the largest centred region of the source with that
    aspect ratio, scaled to fit -- so nothing is squashed. Without, the source is centre-cropped down
    to the nearest multiple of `multiple` and keeps its resolution.
    """
    image = image.convert("RGB")
    if width is None and height is None:
        w, h = (s // multiple * multiple for s in image.size)
        left, top = (image.width - w) // 2, (image.height - h) // 2
        return image.crop((left, top, left + w, top + h))

    assert width is not None and height is not None, "give both width and height, or neither"
    assert (
        width % multiple == 0 and height % multiple == 0
    ), f"{width}x{height} is not a multiple of {multiple}"
    scale = max(width / image.width, height / image.height)
    sw, sh = width / scale, height / scale
    box = ((image.width - sw) / 2, (image.height - sh) / 2, (image.width + sw) / 2, (image.height + sh) / 2)
    return image.resize((width, height), Image.Resampling.LANCZOS, box=box)


def roll(image: Image.Image, dx: int = 0, dy: int = 0) -> Image.Image:
    """The tile cut at a different place: the 1x1 window whose top-left corner sits at (dx, dy) on a
    2x2 board of copies. Equivalent to np.roll by (-dy, -dx); `unroll` puts it back."""
    return Image.fromarray(np.roll(np.asarray(image), shift=(-dy, -dx), axis=(0, 1)))


def unroll(image: Image.Image, dx: int = 0, dy: int = 0) -> Image.Image:
    """Undo `roll`, so a result can be compared with the picture it came from in the same framing.
    For a tile that really is seamless this changes nothing that matters -- every cut of it is as
    good as any other -- but it keeps a chain of passes in one frame of reference."""
    return roll(image, -dx, -dy)


def seam(size: tuple[int, int], dx: int = 0, dy: int = 0) -> tuple[int, int]:
    """Where the tile's own edges ended up after `roll(image, dx, dy)`: the column that was its
    left/right join, and the row that was its top/bottom join. Mask across these and the model
    draws the join the repeat will actually make. At dx = w // 2 the column is the middle one."""
    w, h = size
    return (w - dx) % w, (h - dy) % h


def _empty(size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.zeros((h, w), dtype=bool)


def band(size: tuple[int, int], axis: str, center: int, thickness: int) -> np.ndarray:
    """[h, w] bool: a stripe `thickness` wide centred on `center`, wrapping around the edges.

    axis "v" is a vertical stripe at column `center` (the left/right seam), "h" a horizontal one at
    row `center`. Wrapping matters at offset 0, where the seam sits on the edge and the stripe is
    half at each side -- exactly the two strips that have to agree for the tile to repeat."""
    assert axis in ("v", "h"), axis
    w, h = size
    n = w if axis == "v" else h
    line = (np.arange(n) - (center - thickness // 2)) % n < min(thickness, n)
    mask = _empty(size)
    if axis == "v":
        mask[:, line] = True
    else:
        mask[line, :] = True
    return mask


def cross(size: tuple[int, int], dx: int, dy: int, thickness: int) -> np.ndarray:
    """Both seams of `roll(image, dx, dy)` at once: the one shape whose repair makes all four edges
    join, corners included, in a single pass instead of seamless_fill.py's three."""
    x, y = seam(size, dx, dy)
    return band(size, "v", x, thickness) | band(size, "h", y, thickness)


def rect(size: tuple[int, int], left: int, top: int, width: int, height: int) -> np.ndarray:
    """[h, w] bool: an axis-aligned rectangle, wrapping around the edges like `band`."""
    w, h = size
    mask = _empty(size)
    cols = (np.arange(w) - left) % w < min(width, w)
    rows = (np.arange(h) - top) % h < min(height, h)
    mask[np.ix_(rows, cols)] = True
    return mask


def square(size: tuple[int, int], side: int, center: tuple[int, int] | None = None) -> np.ndarray:
    """A centred square -- what the corner-coupling stage of seamless_fill.py masks."""
    w, h = size
    cx, cy = center if center is not None else (w // 2, h // 2)
    return rect(size, cx - side // 2, cy - side // 2, side, side)


def paint(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = WHITE) -> Image.Image:
    """The picture with everything under `mask` set to one flat colour. White by default: it is the
    one value the model reads as "this is missing" rather than as content to preserve."""
    pixels = np.array(image.convert("RGB"))
    assert pixels.shape[:2] == mask.shape, f"mask {mask.shape} does not fit image {pixels.shape[:2]}"
    pixels[mask] = color
    return Image.fromarray(pixels)


def prepare(
    image: Image.Image,
    dx: int = 0,
    dy: int = 0,
    masks: list[np.ndarray] | np.ndarray | None = None,
    color: tuple[int, int, int] = WHITE,
) -> tuple[Image.Image, np.ndarray]:
    """Roll, then paint: the reference image the model is shown, and the mask that made it.

    This is the bottleneck of the whole workflow. Everything upstream -- which offset, which regions,
    drawn by hand in the web page or built from the helpers above -- exists only to produce this one
    picture, and everything downstream needs nothing else. Save it and a run is reproducible; open it
    and you can see exactly what the model was asked to do.
    """
    rolled = roll(image, dx, dy)
    if masks is None:
        masks = []
    elif isinstance(masks, np.ndarray):
        masks = [masks]
    mask = _empty(rolled.size)
    for m in masks:
        mask |= m
    return paint(rolled, mask, color), mask


def generate_from_reference(
    pipe: TorusPipe,
    prompt: str,
    reference: Image.Image | str | list,
    seed: int | list[int] = 0,
    width: int | None = None,  # default: the reference's own size, so the fill lands where the hole is
    height: int | None = None,
    num_steps: int = 4,  # the distilled klein models' schedule
    guidance: float = 1.0,  # their fixed value; it also drops the unconditional branch (torus.py)
    geo_guidance: float = 2.0,  # 0 for an ordinary, non-repeating edit
    geo_prompt: str = TORUS_PROMPT,
    system_prompt: str | None = None,
    wrap: bool | tuple[bool, bool] = True,
    unanchor_text: bool = False,
    ref_max_pixels: int | None = None,  # default: the reference's own area, i.e. do not shrink it
    **settings,
) -> Tensor:
    """`generate` with no init image and no mask: prompt + reference in, uint8 [P, H, W, 3] out.

    The defaults are the ones that make a masked reference behave. Size follows the reference, and
    so does `ref_max_pixels`: FLUX.2 caps a reference's area and would otherwise quietly resample a
    1024-px tile down to 512, softening the white edges the instruction is pointing at. The cost is
    real -- every 16x16 px of reference is another token written out by hand in torus_attention --
    so drop it for a style reference that does not have to line up with anything.
    """
    first = reference[0] if isinstance(reference, list) and reference else reference
    first = Image.open(first) if isinstance(first, str) else first
    if width is None or height is None:
        assert isinstance(first, Image.Image), "with no reference to follow, give width and height"
        width, height = width or first.width, height or first.height
    return generate(
        pipe,
        prompt,
        seed,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        geo_guidance=geo_guidance,
        geo_prompt=geo_prompt,
        system_prompt=system_prompt,
        wrap=(wrap, wrap) if isinstance(wrap, bool) else wrap,
        unanchor_text=unanchor_text,
        ref_image=reference or None,
        ref_max_pixels=ref_max_pixels or (first.width * first.height if first else 512**2),
        init_image=None,  # the point of this module: no inpainting, no init latent, t_start = 1
        **settings,
    )


def fill(
    pipe: TorusPipe,
    image: Image.Image,
    prompt: str = FILL_PROMPT,
    dx: int = 0,
    dy: int = 0,
    masks: list[np.ndarray] | np.ndarray | None = None,
    keep_framing: bool = True,
    extra_refs: list | None = None,
    **settings,
) -> tuple[Image.Image, Image.Image]:
    """One pass: roll, paint white, generate, roll back. Returns (result, the reference it was shown).

    Chain it by hand for the staged repair seamless_fill.py automates -- a vertical band at
    dx = w // 2, then a horizontal one at dy = h // 2, then a square at both -- or do all of it in
    one pass with `cross`. `keep_framing` undoes the roll so every pass comes back in the same frame.
    """
    tile = crop_to_tile(image)
    reference, _ = prepare(tile, dx, dy, masks)
    refs = [reference, *(extra_refs or [])]
    out = generate_from_reference(pipe, prompt, refs, **settings)[0]
    result = Image.fromarray(out.numpy())
    return (unroll(result, dx, dy) if keep_framing else result), reference
