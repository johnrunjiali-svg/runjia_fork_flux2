"""A one-page live demo of flux2.torus_generate.generate. Standard library only; one GPU, one request at a time.

On the server:    PYTHONPATH=src uv run python scripts/torus_web.py            (CUDA_VISIBLE_DEVICES=3 to pick a GPU)
On your laptop:   ssh -L 7860:localhost:7860 <server>      then open  http://localhost:7860

It listens on 127.0.0.1 only, so the ssh tunnel is the only way in. Every result is also written to
output/web/<time>_{tile,grid,seamless}.png with the request next to it as <time>.json.
"""

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

from flux2.torus_generate import TorusPipe, generate

PAGE = Path(__file__).with_name("torus_web.html")
OUT = Path("output/web")
GPU = threading.Lock()
progress = {"step": 0, "total": 0}


def from_data_url(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def to_data_url(image: torch.Tensor) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image.numpy()).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def three_views(tile: torch.Tensor) -> dict[str, torch.Tensor]:
    """tile: uint8 [H, W, 3]. The tile, 2x2 copies pulled apart so the four are visible, 2x2 copies joined."""
    h, w, _ = tile.shape
    gap = max(4, w // 48)
    grid = torch.full((2 * h + gap, 2 * w + gap, 3), 255, dtype=torch.uint8)
    for top in (0, h + gap):
        for left in (0, w + gap):
            grid[top : top + h, left : left + w] = tile
    return {"tile": tile, "grid": grid, "seamless": tile.repeat(2, 2, 1)}


def run(pipe: TorusPipe, req: dict) -> dict:
    """req: prompt, refs [data url], init (data url, already at the output size) or None,
    keep (gh x gw lists of 0/1, the painted tokens), and the settings of `generate`."""
    s = req["settings"]
    width, height = int(s["width"]), int(s["height"])
    keep = torch.tensor(req["keep"] or [[0]], dtype=torch.uint8)
    # An init image with nothing painted is still worth sending: below t_start = 1 the whole grid
    # starts from it, which is plain image-to-image.
    use_init = req["init"] is not None
    # `generate` keeps a token when its 16x16 pixels are all white, so a mask drawn per token is exact.
    keep_mask = Image.fromarray((keep * 255).repeat_interleave(16, 0).repeat_interleave(16, 1).numpy())

    progress.update(step=0, total=int(s["num_steps"]))
    tile = generate(
        pipe,
        req["prompt"],
        int(s["seed"]),
        width=width,
        height=height,
        num_steps=int(s["num_steps"]),
        guidance=float(s["guidance"]),
        geo_guidance=float(s["geo_guidance"]),
        wrap=(bool(s["wrap"]), bool(s["wrap"])),
        unanchor_text=bool(s["unanchor_text"]),
        ref_image=[from_data_url(r) for r in req["refs"]],
        ref_max_pixels=int(s["ref_max_pixels"]),
        init_image=from_data_url(req["init"]) if use_init else None,
        keep_mask=keep_mask if use_init else None,
        t_start=float(s.get("t_start", 1.0)) if use_init else 1.0,
        on_step=lambda step, total: progress.update(step=step, total=total),
    )[0]

    views = three_views(tile)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    for name, image in views.items():
        Image.fromarray(image.numpy()).save(OUT / f"{stamp}_{name}.png")
    record = {
        "prompt": req["prompt"],
        "settings": s,
        "num_refs": len(req["refs"]),
        "kept_tokens": int(keep.sum()),
    }
    (OUT / f"{stamp}.json").write_text(json.dumps(record, indent=2))
    return {name: to_data_url(image) for name, image in views.items()} | {"saved": str(OUT / stamp)}


def make_handler(pipe: TorusPipe):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, body: bytes, kind: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/progress":
                self.reply(200, json.dumps(progress | {"busy": GPU.locked()}).encode())
            else:
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")  # re-read: edit the page live

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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


def main(model_name: str = "flux.2-klein-base-9b", port: int = 7860):
    # The text encoder stays on the GPU (prompts change with every request): with the 9B model that
    # is 16 + 18 GB on a card without FP8, which leaves a 48 GB A40 room for one image at a time.
    pipe = TorusPipe(model_name)
    print(f"ready:  ssh -L {port}:localhost:{port} <this server>   then open   http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(pipe)).serve_forever()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
