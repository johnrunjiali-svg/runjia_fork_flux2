"""Step 0, on the laptop: bring every test pattern to one size, 1024 x 1024.

    uv run python scripts/resize_tiles.py
    uv run python scripts/resize_tiles.py --src some/folder --dst principled/tiles_512 --size 512

Every image under --src (subfolders included) is resampled as one period of a repeat, so the resize
neither adds a seam nor hides the one that is there (seamless.tiles.resize_periodic). The copies get
plain ASCII names -- `00_最初玫瑰样例_Original_Rose_Sample.png` becomes `00_Original_Rose_Sample.png`
-- because they are about to be typed on a command line and carried over ssh; tiles.csv says which
source each one came from. Upload the result and nothing else:

    rsync -av principled/tiles_1024 <server>:<repo>/principled/
"""

import csv
import re
from pathlib import Path

from PIL import Image

from seamless.tiles import resize_periodic, seam_jump

IMAGES = {".png", ".jpg", ".jpeg", ".webp"}


def ascii_name(stem: str) -> str:
    """What is left of a file name once everything outside [A-Za-z0-9_-] is dropped."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]", "", stem)).strip("_")


def main(src: str = "principled/test_patterns", dst: str = "principled/tiles_1024", size: int = 1024):
    assert size % 16 == 0, f"{size} is not a multiple of 16, the size of one FLUX.2 token"
    sources = sorted(p for p in Path(src).rglob("*") if p.suffix.lower() in IMAGES)
    assert sources, f"no images under {src}"
    names = [ascii_name(p.stem) or f"tile_{k:02d}" for k, p in enumerate(sources)]
    assert len(set(names)) == len(names), f"two sources share a name: {sorted(names)}"

    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, name in sorted(zip(sources, names), key=lambda pair: pair[1]):
        image = Image.open(path)
        tile = resize_periodic(image, size)
        tile.save(out / f"{name}.png")
        rows.append([f"{name}.png", str(path), f"{image.width}x{image.height}", f"{seam_jump(tile):.2f}"])
        print(f"{image.width}x{image.height} -> {size}x{size}  seam {rows[-1][3]:>5}  {name}.png")
    with (out / "tiles.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["file", "source", "source_size", "seam_jump"], *rows])
    print(
        f"{len(rows)} tiles in {out}/   (seam: 1 = the edges join like any other line, see tiles.seam_jump)"
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
