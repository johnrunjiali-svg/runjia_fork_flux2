"""One picture per run that holds every step at full resolution, with the lines that matter marked.

Every panel is pasted at its own pixel size -- nothing is resampled, so zooming into the sheet is
zooming into the PNG of that step. Single tiles go in the top row in the order they were made, their
2x2 repeats in the row below. Everything drawn by this file sits in the margin around a panel and
never on it:

  red arrows    a seam: the line where the tile's own edges meet. In a rolled picture it runs
                through the middle; in a 2x2 repeat it is where the copies touch.
  blue arrows   where the edges of the *generated* picture meet once it is rolled back. Nothing
                was masked here, so the line shows whether the model's output closes up on itself:
                with the wrapped attention it draws the two sides as neighbours, without it apart.
  amber bars    the extent of the hole: what was white in the model's input, and so what is new in
                its output.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (24, 24, 27)  # dark, so a white hole at the edge of a panel still has an outline
TEXT, DIM = (236, 236, 236), (160, 160, 165)
SEAM, WRAP, HOLE = (255, 76, 76), (70, 170, 255), (255, 196, 40)

EDGE = 48  # around the whole sheet
MARGIN = 128  # around every panel: bars, then arrows
TITLE = 64  # above every panel
BAR = (8, 20)  # a hole bar spans this range of distances from the panel
ARROW = (30, 96)  # an arrow's tip and tail, likewise
HEAD, SHAFT = (28, 22), 6  # arrow head length and half width, shaft half width


@dataclass
class Panel:
    name: str  # file stem; run_tile prefixes its place in the pipeline: "3_rolled"
    title: str
    image: Image.Image
    copies: int = 1  # 2 for a 2x2 repeat
    offset: tuple[int, int] = (0, 0)  # how far the picture is rolled from the tile's own framing
    marks: str = ""  # which of "seam", "wrap", "hole" to draw around it
    tag: str = ""  # "model input" or "result"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size)  # Pillow's built-in scalable font: the same on every machine


def seam_lines(length: int, period: int, shift: int) -> list[int]:
    """The lines strictly inside a picture `length` px long that are a seam of its `period`-px tile,
    when the picture is rolled `shift` px from the tile's own framing."""
    return [c for c in range(1, length) if (c + shift) % period == 0]


def hole_spans(length: int, period: int, shift: int, hole: int) -> list[tuple[int, int]]:
    """[start, end) runs of pixels within hole / 2 of a seam, edges of the picture included."""
    inside = (np.arange(length) + shift + hole // 2) % period < hole
    steps = np.flatnonzero(np.diff(np.concatenate(([0], inside.astype(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(steps[::2], steps[1::2])]


def arrow(draw: ImageDraw.ImageDraw, tip: tuple[int, int], direction: tuple[int, int], color):
    """An arrow whose tip is at `tip` and which points along `direction`, a unit axis vector."""
    (x, y), (dx, dy) = tip, direction
    length = ARROW[1] - ARROW[0]

    def at(along: float, across: float) -> tuple[float, float]:  # `along` runs back from the tip
        return x - dx * along - dy * across, y - dy * along + dx * across

    draw.polygon([at(0, 0), at(HEAD[0], HEAD[1]), at(HEAD[0], -HEAD[1])], fill=color)
    draw.polygon([at(HEAD[0], SHAFT), at(length, SHAFT), at(length, -SHAFT), at(HEAD[0], -SHAFT)], fill=color)


def mark(draw: ImageDraw.ImageDraw, panel: Panel, left: int, top: int, hole: int, generated_offset):
    """Arrows and bars in the margin of a panel whose top-left pixel is at (left, top)."""
    w, h = panel.image.size
    right, bottom = left + w, top + h
    period = (w // panel.copies, h // panel.copies)
    if "hole" in panel.marks:
        for a, b in hole_spans(w, period[0], panel.offset[0], hole):
            draw.rectangle((left + a, top - BAR[1], left + b - 1, top - BAR[0] - 1), fill=HOLE)
            draw.rectangle((left + a, bottom + BAR[0], left + b - 1, bottom + BAR[1] - 1), fill=HOLE)
        for a, b in hole_spans(h, period[1], panel.offset[1], hole):
            draw.rectangle((left - BAR[1], top + a, left - BAR[0] - 1, top + b - 1), fill=HOLE)
            draw.rectangle((right + BAR[0], top + a, right + BAR[1] - 1, top + b - 1), fill=HOLE)

    seams = [
        seam_lines(n, p, o) if "seam" in panel.marks else [] for n, p, o in zip((w, h), period, panel.offset)
    ]
    wraps = [
        [c for c in seam_lines(n, p, o - g) if c not in seen] if "wrap" in panel.marks else []
        for n, p, o, g, seen in zip((w, h), period, panel.offset, generated_offset, seams)
    ]
    for (xs, ys), color in ((seams, SEAM), (wraps, WRAP)):
        for x in xs:
            arrow(draw, (left + x, top - ARROW[0]), (0, 1), color)
            arrow(draw, (left + x, bottom + ARROW[0]), (0, -1), color)
        for y in ys:
            arrow(draw, (left - ARROW[0], top + y), (1, 0), color)
            arrow(draw, (right + ARROW[0], top + y), (-1, 0), color)


def wrap_text(text: str, face: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and face.getlength(f"{line} {word}") > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line]


def build_sheet(
    panels: list[Panel],
    hole: int,
    header: list[str],
    generated_offset: tuple[int, int] = (0, 0),
) -> Image.Image:
    """`header`: a title line, then as many lines of small print as needed. `hole`: thickness of the
    white region across a seam. `generated_offset`: how far the model's output is rolled from the
    tile's own framing, which is where its edges are."""
    rows = [[p for p in panels if p.copies == 1], [p for p in panels if p.copies > 1]]
    rows = [row for row in rows if row]
    row_width = [sum(p.image.width + 2 * MARGIN for p in row) for row in rows]
    row_height = [max(p.image.height for p in row) + 2 * MARGIN + TITLE for row in rows]
    width = max(row_width) + 2 * EDGE

    big, small, caption = font(72), font(42), font(38)
    text = [(header[0], big, TEXT)]
    text += [(line, small, DIM) for para in header[1:] for line in wrap_text(para, small, width - 2 * EDGE)]
    line_height = {id(big): 98, id(small): 58}
    legend_height = 84
    header_height = sum(line_height[id(face)] for _, face, _ in text) + legend_height

    sheet = Image.new("RGB", (width, 2 * EDGE + header_height + sum(row_height)), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    y = EDGE
    for line, face, color in text:
        draw.text((EDGE, y), line, font=face, fill=color)
        y += line_height[id(face)]

    x = EDGE + 8  # the legend: the three marks, drawn as they are drawn around the panels
    middle = y + legend_height // 2
    for color, label in (
        (SEAM, "seam: the tile's own edges meet here"),
        (WRAP, "the generated picture's edges meet here"),
        (HOLE, "extent of the hole the model fills"),
    ):
        if color == HOLE:
            draw.rectangle((x, middle - 6, x + 66, middle + 5), fill=color)
        else:
            arrow(draw, (x + 66, middle), (1, 0), color)
        draw.text((x + 86, middle), label, font=small, fill=TEXT, anchor="lm")
        x += 86 + int(small.getlength(label)) + 72

    top = EDGE + header_height
    for row, height in zip(rows, row_height):
        left = EDGE
        for panel in row:
            px, py = left + MARGIN, top + TITLE + MARGIN
            sheet.paste(panel.image, (px, py))
            mark(draw, panel, px, py, hole, generated_offset)
            draw.text((px, top + TITLE // 2), panel.name, font=caption, fill=TEXT, anchor="lm")
            cursor = px + caption.getlength(panel.name + "   ")
            draw.text((cursor, top + TITLE // 2), panel.title, font=caption, fill=DIM, anchor="lm")
            if panel.tag:
                cursor += caption.getlength(panel.title + "   ")
                draw.text((cursor, top + TITLE // 2), f"[{panel.tag}]", font=caption, fill=HOLE, anchor="lm")
            left += panel.image.width + 2 * MARGIN
        top += height
    return sheet
