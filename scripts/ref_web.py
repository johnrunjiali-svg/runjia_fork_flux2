"""Roll, mask, generate: the image-to-image workflow of flux2.masked_ref, as one live page.

Two stages that meet at one file. The studio (all of it in the browser, no GPU) takes a picture,
rolls it -- a 1x1 window slid over a 2x2 board of copies, which brings any seam into the open -- and
lets you paint regions pure white. That masked picture is the bottleneck: it is the only thing the
generator is given, it is downloadable, and a saved one can be dropped straight back in later. The
generator then runs plain FLUX.2 text+image-to-image on it: no mask channel, no init latent, no
inpainting model. Why white instead of inpainting is in src/flux2/masked_ref.py.

On the server:    PYTHONPATH=src uv run python scripts/ref_web.py            (CUDA_VISIBLE_DEVICES=3 to pick a GPU)
                  ... --model_name flux.2-klein-base-9b                      (undistilled: guidance above 1)
On your laptop:   ssh -L 7861:localhost:7861 <server>      then open  http://localhost:7861

It listens on 127.0.0.1 only, so the ssh tunnel is the only way in. Every run writes a folder under
output/ref_web/<time>/ with the reference, the mask, the source, every view of the result and a
run.json; "Save run" in the page zips one folder and "Save session" zips all of them.
"""

import base64
import io
import json
import re
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

from flux2.masked_ref import FILL_PROMPT, SYSTEM_PRESETS, generate_from_reference, unroll
from flux2.torus import TORUS_PROMPT
from flux2.torus_generate import TorusPipe

PAGE = Path(__file__).with_name("ref_web.html")
OUT = Path("output/ref_web")
STAMP = re.compile(r"^\d{4}_\d{6}$")  # what `run` names a folder; nothing else may be zipped
GPU = threading.Lock()
progress = {"step": 0, "total": 0}
runs: list[str] = []  # the folders this process wrote, newest last


def from_data_url(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def views(tile: Image.Image, dx: int, dy: int) -> dict[str, Image.Image]:
    """What the page can show: the tile, the tile rolled back to the framing the studio started
    from, 2x2 copies pulled apart so all four are visible, and 2x2 copies joined."""
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


def run(pipe: TorusPipe, req: dict) -> dict:
    """req: prompt, system_prompt, geo_prompt, reference (the masked picture, a data url, or None
    for plain text-to-image), refs (extra references), source and mask (kept for the record only),
    roll {dx, dy}, and the settings of `generate`."""
    s = req["settings"]
    reference = from_data_url(req["reference"]) if req.get("reference") else None
    extra = [from_data_url(r) for r in req.get("refs") or []]
    dx, dy = int(req.get("roll", {}).get("dx", 0)), int(req.get("roll", {}).get("dy", 0))

    progress.update(step=0, total=int(s["num_steps"]))
    started = time.time()
    tile = generate_from_reference(
        pipe,
        req["prompt"],
        [reference, *extra] if reference else extra,
        seed=int(s["seed"]),
        width=int(s["width"]),
        height=int(s["height"]),
        num_steps=int(s["num_steps"]),
        guidance=float(s["guidance"]),
        geo_guidance=float(s["geo_guidance"]),
        geo_prompt=req.get("geo_prompt") or TORUS_PROMPT,
        system_prompt=req.get("system_prompt") or None,
        wrap=(bool(s["wrap"]), bool(s["wrap"])),
        unanchor_text=bool(s["unanchor_text"]),
        ref_max_pixels=int(s["ref_max_pixels"]),
        on_step=lambda step, total: progress.update(step=step, total=total),
    )[0]
    seconds = time.time() - started

    out = views(Image.fromarray(tile.numpy()), dx, dy)
    stamp = time.strftime("%m%d_%H%M%S")
    folder = OUT / stamp
    folder.mkdir(parents=True, exist_ok=True)
    for name, image in out.items():
        image.save(folder / f"{name}.png")
    if reference is not None:
        reference.save(folder / "reference.png")  # the bottleneck: enough on its own to repeat the run
    for k, image in enumerate(extra):
        image.save(folder / f"ref{k + 2}.png")
    for name in ("source", "mask"):
        if req.get(name):
            from_data_url(req[name]).save(folder / f"{name}.png")
    (folder / "run.json").write_text(
        json.dumps(
            {
                "prompt": req["prompt"],
                "system_prompt": req.get("system_prompt") or None,
                "geo_prompt": req.get("geo_prompt") or TORUS_PROMPT,
                "settings": s,
                "roll": {"dx": dx, "dy": dy},
                "masked_fraction": req.get("masked_fraction"),
                "num_extra_refs": len(extra),
                "model_name": pipe.model_name,
                "seconds": round(seconds, 1),
            },
            indent=2,
        )
    )
    runs.append(stamp)
    return {name: to_data_url(image) for name, image in out.items()} | {
        "run": stamp,
        "saved": str(folder),
        "seconds": round(seconds, 1),
    }


def zip_runs(stamps: list[str]) -> bytes:
    """One folder or all of them, as a zip the browser downloads. Only folders this process wrote,
    so a stamp coming from the page can never name a path of its own."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for stamp in stamps:
            assert STAMP.match(stamp) and stamp in runs, f"unknown run {stamp!r}"
            for path in sorted((OUT / stamp).rglob("*")):
                if path.is_file():
                    archive.write(path, f"{stamp}/{path.relative_to(OUT / stamp)}")
    return buffer.getvalue()


def make_handler(pipe: TorusPipe):
    info = json.dumps(
        {
            "model_name": pipe.model_name,
            "distilled": pipe.distilled,
            "defaults": pipe.defaults,  # what this model was distilled for: the page starts there
            "fill_prompt": FILL_PROMPT,
            "geo_prompt": TORUS_PROMPT,
            "system_presets": SYSTEM_PRESETS,
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, body: bytes, kind: str = "application/json", filename: str = ""):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/progress":
                self.reply(200, json.dumps(progress | {"busy": GPU.locked()}).encode())
            elif self.path == "/info":
                self.reply(200, info)
            else:
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")  # re-read: edit the page live

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/save":
                stamps = runs[:] if req.get("all") else [req["run"]]
                if not stamps:
                    return self.reply(404, json.dumps({"error": "nothing to save yet"}).encode())
                try:
                    name = f"ref_web_{stamps[0]}{'_plus' if len(stamps) > 1 else ''}.zip"
                    return self.reply(200, zip_runs(stamps), "application/zip", name)
                except AssertionError as e:
                    return self.reply(404, json.dumps({"error": str(e)}).encode())
            if not GPU.acquire(blocking=False):
                return self.reply(409, json.dumps({"error": "the GPU is busy with another request"}).encode())
            try:
                self.reply(200, json.dumps(run(pipe, req)).encode())
            except Exception as e:  # noqa: BLE001  the page shows it; the models stay loaded
                torch.cuda.empty_cache()
                self.reply(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())
            finally:
                GPU.release()

        def log_message(self, *args):  # /progress is polled twice a second
            pass

    return Handler


def main(model_name: str = "flux.2-klein-9b", port: int = 7861):
    # The text encoder stays on the GPU: prompts, and now system prompts, change with every request.
    pipe = TorusPipe(model_name)
    print(f"ready:  ssh -L {port}:localhost:{port} <this server>   then open   http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(pipe)).serve_forever()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
