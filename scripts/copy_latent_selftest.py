"""Seconds on a CPU, random toy weights, no checkpoint:  uv run python scripts/copy_latent_selftest.py

1. The chop and the copy, with a stand-in for the model that returns a number per token that says
   where the token sat (h * 1000 + w): after one step the core moved by exactly the velocity of its
   own window of the wide grid, and the model was handed ids on [-band, g + band)^2.
2. What the model is handed: step 1 is the noise padded circularly (copy (i, j) is core token
   (i mod gh, j mod gw)), and step 2 is the *moved* core padded again, not the old copies carried on.
3. band 0 is the stock sampler, `sampling.denoise`; band 2 is not. Guidance != 1 with the empty
   prompt set equal to the prompt is guidance 1.
4. seamless.copy_latent end to end on toy weights: a picture of the right size, deterministic, the
   stock column equal to experiment 4's band 0, and every other column a different picture; then a
   whole job through run_job and the report.
"""

import json
import tempfile
from pathlib import Path

import torch
from einops import rearrange
from torch.nn import functional as F

from flux2.copy_latent import denoise_copy_latent, padded_ids
from flux2.model import Flux2
from flux2.sampling import denoise
from seamless.copy_latent import KleinLatent, run_job, variants, write_report
from seamless.copy_tokens import KleinCopy
from torus_selftest import ToyParams, make_ids


class Spy:
    """A model that returns h * 1000 + w for every token of the ids it is handed, and remembers its inputs."""

    def __init__(self):
        self.seen = []

    def __call__(self, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        self.seen.append((x.clone(), x_ids.clone()))
        return (x_ids[..., 1] * 1000 + x_ids[..., 2])[..., None].expand_as(x).to(x.dtype)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    gh, gw, band, num_txt, c = 4, 6, 2, 5, 8
    core = torch.randn(1, c, gh, gw)
    txt = torch.randn(1, num_txt, 12)
    _, txt_ids = make_ids(gh, gw, num_txt)
    timesteps = [1.0, 0.6, 0.0]

    # 1
    spy = Spy()
    out = denoise_copy_latent(spy, core, txt, txt_ids, timesteps[:2], band)
    window = torch.tensor([[h * 1000 + w for w in range(gw)] for h in range(gh)], dtype=core.dtype)
    assert torch.allclose(out, core + (0.6 - 1.0) * window), "the core did not move by the velocity of its own window"
    ids = spy.seen[0][1]
    assert ids.shape == (1, (gh + 2 * band) * (gw + 2 * band), 4)
    assert ids[0, :, 1].min() == -band and ids[0, :, 1].max() == gh + band - 1
    assert ids[0, :, 2].min() == -band and ids[0, :, 2].max() == gw + band - 1
    assert (ids == padded_ids((gh, gw), band, "cpu")[None]).all()
    print("ok   the velocity of the middle window moves the core; ids span [-band, g + band)")

    # 2
    spy = Spy()
    denoise_copy_latent(spy, core, txt, txt_ids, timesteps, band)
    for step, state in enumerate((core, core + (0.6 - 1.0) * window)):
        x, ids = spy.seen[step]
        for token, (_, i, j, _) in zip(x[0], ids[0].tolist()):
            assert torch.equal(token, state[0, :, i % gh, j % gw]), (step, i, j)
    stale = rearrange(F.pad(core, (band,) * 4, mode="circular"), "b c h w -> b (h w) c")
    assert not torch.equal(spy.seen[1][0], stale)
    print("ok   every step's input is the current core padded circularly; copies are not carried over")

    # 3
    model = Flux2(ToyParams()).eval()
    ids_1, ctx_ids = make_ids(gh, gw, num_txt)
    noise = torch.randn(1, c, gh, gw)
    stock = denoise(model, rearrange(noise, "b c h w -> b (h w) c"), ids_1, txt, ctx_ids, timesteps, 1.0)
    got = {b: denoise_copy_latent(model, noise, txt, ctx_ids, timesteps, b) for b in (0, 2)}
    err = (rearrange(got[0], "b c h w -> b (h w) c") - stock).abs().max().item()
    assert err < 1e-5, err
    assert (got[2] - got[0]).abs().max() > 1e-3, "band 2 changed nothing"
    cfg = denoise_copy_latent(model, noise, torch.cat((txt, txt)), torch.cat((ctx_ids, ctx_ids)), timesteps, 2, guidance=3.0)
    assert torch.allclose(cfg, got[2], atol=1e-5), "CFG with the empty prompt equal to the prompt is not guidance 1"
    print(f"ok   band 0 equals the stock sampler ({err:.1e}), band 2 does not; CFG reduces to guidance 1")

    # 4
    klein = KleinLatent.toy(["moss", "roses"], num_steps=2)
    pic = {(m, b): klein.generate("moss", 0, m, b, 160) for m, b in [("latent", 0), ("latent", 2), ("tokens", 2), ("rope", None)]}
    assert all(p.size == (160, 160) and p.mode == "RGB" for p in pic.values())
    assert pic["latent", 2].tobytes() == klein.generate("moss", 0, "latent", 2, 160).tobytes(), "same seed, different picture"
    assert klein.generate("moss", 1, "latent", 2, 160).tobytes() != pic["latent", 2].tobytes(), "the seed changed nothing"
    experiment_4 = KleinCopy.__call__(klein, "moss", 0, 0, None, 160)
    diff = (torch.tensor(list(pic["latent", 0].tobytes())).float() - torch.tensor(list(experiment_4.tobytes())).float()).abs().max()
    assert diff <= 3, f"stock column differs from experiment 4's band 0 by {diff}"
    assert len({p.tobytes() for p in pic.values()}) == 4, "two columns are the same picture"
    print(f"ok   KleinLatent: 160 px pictures, deterministic, stock == experiment 4 band 0 (max {diff:.0f}/255), 4 columns differ")

    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "bands": [0, 2], "controls": ["tokens", "rope"], "size": 160, "seed": 0, "num_steps": 2, "guidance": 1.0,
            "model_name": "toy", "system_prompt": "", "toy": True, "commit": None,
            "jobs": [{"name": "00_moss", "prompt": "moss"}],
        }
        assert [v[0] for v in variants(config)] == ["stock", "latent2", "tokens2", "rope"]
        Path(tmp, "config.json").write_text(json.dumps(config))
        run_job(config["jobs"][0], tmp, config, klein)
        folder = Path(tmp, "00_moss")
        assert len(list(folder.glob("*.png"))) == 4 * 2 + 1 and (folder / "thumb.jpg").exists() and (folder / "run.json").exists()
        index = write_report(tmp)
        assert "latent2" in index.read_text() and "latent2" in Path(tmp, "summary.csv").read_text()
    print("ok   run_job and the report: 4 columns x (result, 2x2) + sheet, index.html, summary.csv")
    print("all passed")


if __name__ == "__main__":
    main()
