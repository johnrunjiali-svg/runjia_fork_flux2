"""Roll, mask or outline, generate: the image-to-image workflow of qwen_torus, as one live page.

Two stages that meet at one file. The studio (all of it in the browser, no GPU) takes a picture, rolls
it -- a 1x1 window slid over a 2x2 board of copies, which brings any seam into the open -- and lets
you either paint regions pure white or draw coloured outlines around them. That annotated picture is
the bottleneck: it is the only thing the generator is given, it is downloadable, and a saved one can
be dropped straight back in later. The generator runs Qwen-Image 2.1 text+image-to-image on it with
the target image on a torus (src/qwen_torus): "fill the white", or "inside the red outline, ...".

On the server:    PYTHONPATH=src uv run python scripts/qwen_web.py             (CUDA_VISIBLE_DEVICES=3 to pick a GPU)
                  ... --vae_tiling                                             (for 2048^2 outputs)
On your laptop:   ssh -L 7862:localhost:7862 <server>      then open  http://localhost:7862
Without a GPU:    PYTHONPATH=src uv run python scripts/qwen_web.py --fake       (toy weights; checks the page, not the pictures)

It listens on 127.0.0.1 only, so the ssh tunnel is the only way in. Every run writes a folder under
output/qwen_web/<time>/ with the reference, the mask, the outlines, the source, every view of the
result and a run.json; "Save run" in the page zips one folder and "Save session" zips all of them.
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

from qwen_torus import FILL_PROMPT, OUTLINE_COLORS, OUTLINE_PROMPT, TORUS_PROMPT, PeConfig, generate, views

PAGE = Path(__file__).with_name("qwen_web.html")
OUT = Path("output/qwen_web")
STAMP = re.compile(r"^\d{4}_\d{6}$")  # what `run` names a folder; nothing else may be zipped
GPU = threading.Lock()
progress = {"step": 0, "total": 0}
runs: list[str] = []  # the folders this process wrote, newest last

# Where the page starts. Qwen-Image 2.1 is sampled without guidance by its authors (prompt strength 1,
# which drops the unconditional branch); the seamless strength is untuned, as it was for FLUX.2.
DEFAULTS = {"num_steps": 50, "guidance": 1.0, "geo_guidance": 2.0}


def from_data_url(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def run(pipe, req: dict) -> dict:
    """req: prompt, negative_prompt, geo_prompt, reference (the annotated picture, a data url, or None
    for plain text-to-image), refs (extra references), source, mask and lines (kept for the record
    only), roll {dx, dy}, outlines (the colours drawn), and settings including `pe` (PeConfig fields)."""
    s = req["settings"]
    reference = from_data_url(req["reference"]) if req.get("reference") else None
    extra = [from_data_url(r) for r in req.get("refs") or []]
    dx, dy = int(req.get("roll", {}).get("dx", 0)), int(req.get("roll", {}).get("dy", 0))
    pe = PeConfig.from_dict(s.get("pe"))

    progress.update(step=0, total=int(s["num_steps"]))
    started = time.time()
    tile = generate(
        pipe,
        req["prompt"],
        seed=int(s["seed"]),
        width=int(s["width"]),
        height=int(s["height"]),
        num_steps=int(s["num_steps"]),
        guidance=float(s["guidance"]),
        geo_guidance=float(s["geo_guidance"]),
        geo_prompt=req.get("geo_prompt") or TORUS_PROMPT,
        negative_prompt=req.get("negative_prompt") or "",
        pe=pe,
        cond_images=([reference, *extra] if reference else extra) or None,
        ref_max_pixels=int(s["ref_max_pixels"]),
        on_step=lambda step, total: progress.update(step=step, total=total),
    )
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
    for name in ("source", "mask", "lines"):
        if req.get(name):
            Image.open(io.BytesIO(base64.b64decode(req[name].split(",", 1)[1]))).save(folder / f"{name}.png")
    (folder / "run.json").write_text(
        json.dumps(
            {
                "prompt": req["prompt"],
                "negative_prompt": req.get("negative_prompt") or "",
                "geo_prompt": req.get("geo_prompt") or TORUS_PROMPT,
                "settings": {**s, "pe": pe.to_dict()},
                "roll": {"dx": dx, "dy": dy},
                "masked_fraction": req.get("masked_fraction"),
                "outlines": req.get("outlines") or [],
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


def make_handler(pipe):
    info = json.dumps(
        {
            "model_name": pipe.model_name,
            "defaults": DEFAULTS,
            "fill_prompt": FILL_PROMPT,
            "outline_prompt": OUTLINE_PROMPT,
            "outline_colors": OUTLINE_COLORS,
            "geo_prompt": TORUS_PROMPT,
            "pe": PeConfig().to_dict(),
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
                    name = f"qwen_web_{stamps[0]}{'_plus' if len(stamps) > 1 else ''}.zip"
                    return self.reply(200, zip_runs(stamps), "application/zip", name)
                except AssertionError as e:
                    return self.reply(404, json.dumps({"error": str(e)}).encode())
            if not GPU.acquire(blocking=False):
                return self.reply(409, json.dumps({"error": "the GPU is busy with another request"}).encode())
            try:
                self.reply(200, json.dumps(run(pipe, req)).encode())
            except Exception as e:  # noqa: BLE001  the page shows it; the models stay loaded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.reply(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())
            finally:
                GPU.release()

        def log_message(self, *args):  # /progress is polled twice a second
            pass

    return Handler


def main(
    model_id: str = "Qwen/Qwen-Image-2.1",
    port: int = 7862,
    cpu_offload: bool = True,  # text encoder, transformer and VAE take turns on the GPU; off = all resident
    vae_tiling: bool = False,
    fake: bool = False,  # toy weights on the CPU: exercises the page and the sampler, draws noise
):
    if fake:
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from qwen_selftest import FakePipe

        pipe = FakePipe()
    else:
        from qwen_torus import QwenTorusPipe

        pipe = QwenTorusPipe(model_id, cpu_offload=cpu_offload, vae_tiling=vae_tiling)
    print(f"ready:  ssh -L {port}:localhost:{port} <this server>   then open   http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(pipe)).serve_forever()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
