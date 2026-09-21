"""Seconds on a CPU, random toy weights, no checkpoint:  PYTHONPATH=src python scripts/torus_selftest.py

1. torus_attention == attention written pair by pair from the nearest-copy displacement.
2. wrap off: torus_forward == the stock Flux2.forward.
3. unanchor_text: rolling the input latent on the torus rolls the output, i.e. the network cannot
   tell where the image was cut. With the text anchored (the default) the same test must fail.
4. decode_torus returns the right size.
5. With reference tokens ([txt, img, ref], ref grid larger than the image grid): 1 and 2 again.
6. denoise_torus with `keep`: kept tokens come out exactly clean, free tokens do not -- from pure
   noise, and from a partially noised start (t_start) with the noise handed in separately.
7. schedule_from: starts at exactly t_start, ends at 0, strictly decreasing, t_start=1 is stock.
"""

import itertools
from dataclasses import dataclass, field

import torch
from einops import rearrange

from flux2.autoencoder import AutoEncoder, AutoEncoderParams
from flux2.model import Flux2, rope
from flux2.sampling import get_schedule
from flux2.torus import build_torus_geometry, decode_torus, denoise_torus, torus_attention, torus_forward
from flux2.torus_generate import schedule_from


@dataclass
class ToyParams:
    in_channels: int = 8
    context_in_dim: int = 12
    hidden_size: int = 256  # 2 heads of 128: the head dim has to stay sum(axes_dim)
    num_heads: int = 2
    depth: int = 2
    depth_single_blocks: int = 2
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


def make_ids(gh: int, gw: int, num_txt: int):
    zero = torch.arange(1)
    x_ids = torch.cartesian_prod(zero, torch.arange(gh), torch.arange(gw), zero)[None]
    ctx_ids = torch.cartesian_prod(zero, zero, zero, torch.arange(num_txt))[None]
    return x_ids, ctx_ids


def make_ref_ids(rh: int, rw: int):
    zero = torch.arange(1)
    return torch.cartesian_prod(zero + 10, torch.arange(rh), torch.arange(rw), zero)[None]


def roll(tokens, gh, shift):
    """Cyclic shift of a [B, gh * gw, C] token sequence on its grid."""
    grid = torch.roll(rearrange(tokens, "b (h w) c -> b h w c", h=gh), shift, (1, 2))
    return rearrange(grid, "b h w c -> b (h w) c")


def pairwise_attention(q, k, v, ids, num_txt, grid, wrap, unanchor_text, theta=2000):
    """The definition, with no trick: logit[p, q] = sum over axes of x_p^T R(displacement) x_q.
    Tokens are [txt, img, ref]; the img tokens are the grid[0] * grid[1] right after the text."""
    n_tok = ids.shape[1]
    index = torch.arange(n_tok)
    is_img = (index >= num_txt) & (index < num_txt + grid[0] * grid[1])
    img_img = is_img[:, None] & is_img[None, :]
    txt_pair = (index < num_txt)[:, None] | (index < num_txt)[None, :]
    logits = 0
    for axis in range(4):
        pos = ids[0, :, axis]
        d = pos[None, :] - pos[:, None]
        if axis in (1, 2):
            n = grid[axis - 1]
            if wrap[axis - 1]:
                d = torch.where(img_img, (d + n // 2) % n - n // 2, d)
            if unanchor_text:
                d = torch.where(txt_pair, 0, d)
        rot = rope(d, 32, theta).double()  # [N, N, 16, 2, 2]
        qa = q[..., 32 * axis : 32 * axis + 32].reshape(*q.shape[:-1], 16, 2).double()
        ka = k[..., 32 * axis : 32 * axis + 32].reshape(*k.shape[:-1], 16, 2).double()
        logits = logits + torch.einsum("bhpmi,pqmij,bhqmj->bhpq", qa, rot, ka)
    attn = torch.softmax(logits * q.shape[-1] ** -0.5, dim=-1) @ v.double()
    return rearrange(attn, "b h n d -> b n (h d)")


@torch.no_grad()
def main():
    torch.manual_seed(0)
    model = Flux2(ToyParams()).eval()
    num_txt = 5

    for (gh, gw), wrap, unanchor_text in itertools.product(
        [(6, 8), (5, 7), (2, 3)], [(True, True), (False, True), (False, False)], [False, True]
    ):
        x_ids, ctx_ids = make_ids(gh, gw, num_txt)
        geo = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), wrap, unanchor_text)
        tag = f"grid {gh}x{gw} wrap {wrap} unanchor_text {unanchor_text}"

        # 1
        q, k, v = torch.randn(3, 2, 2, num_txt + gh * gw, 128).unbind(0)
        got = torus_attention(q, k, v, geo, q_chunk=7)
        want = pairwise_attention(
            q, k, v, torch.cat((ctx_ids, x_ids), 1), num_txt, (gh, gw), wrap, unanchor_text
        )
        err = (got - want).abs().max().item()
        assert err < 1e-5, (tag, err)

        x = torch.randn(2, gh * gw, 8)
        ctx = torch.randn(2, num_txt, 12)
        t = torch.tensor([0.7, 0.3])
        out = torus_forward(model, x, t, ctx, None, geo, q_chunk=7)

        # 2
        if wrap == (False, False) and not unanchor_text:
            stock = model(x, x_ids.expand(2, -1, -1), t, ctx, ctx_ids.expand(2, -1, -1), None)
            err = (out - stock).abs().max().item()
            assert err < 1e-4, (tag, err)
            print(f"ok   {tag}: equals stock forward ({err:.1e})")

        # 3
        if wrap == (True, True):
            worst = 0.0
            for shift in [(1, 0), (0, 1), (gh // 2, gw // 2), (gh - 1, 3)]:
                rolled_out = torus_forward(model, roll(x, gh, shift), t, ctx, None, geo, q_chunk=7)
                worst = max(worst, (rolled_out - roll(out, gh, shift)).abs().max().item())
            if unanchor_text:
                assert worst < 1e-4, (tag, worst)
                print(f"ok   {tag}: roll-equivariant ({worst:.1e})")
            else:
                assert worst > 1e-5, (tag, worst)
                print(f"ok   {tag}: NOT roll-equivariant, the text marks the origin ({worst:.1e})")

    # 5
    gh, gw = 6, 8
    x_ids, ctx_ids = make_ids(gh, gw, num_txt)
    ref_ids = make_ref_ids(7, 11)
    n_ref = ref_ids.shape[1]
    x_ref = torch.randn(2, gh * gw + n_ref, 8)
    for wrap, unanchor_text in itertools.product([(True, True), (False, False)], [False, True]):
        geo = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), wrap, unanchor_text, ref_ids)
        q, k, v = torch.randn(3, 2, 2, num_txt + gh * gw + n_ref, 128).unbind(0)
        ids = torch.cat((ctx_ids, x_ids, ref_ids), 1)
        want = pairwise_attention(q, k, v, ids, num_txt, (gh, gw), wrap, unanchor_text)
        err = (torus_attention(q, k, v, geo, q_chunk=7) - want).abs().max().item()
        assert err < 1e-5, ("ref", wrap, unanchor_text, err)
        if wrap == (False, False) and not unanchor_text:
            out = torus_forward(model, x_ref, t, ctx, None, geo)
            img_ids = torch.cat((x_ids, ref_ids), 1).expand(2, -1, -1)
            stock = model(x_ref, img_ids, t, ctx, ctx_ids.expand(2, -1, -1), None)
            err = (out - stock).abs().max().item()
            assert err < 1e-4, ("ref stock", err)
    print("ok   reference tokens: pairwise attention, and wrap off equals stock forward with refs")

    # 6
    geo = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), (True, True), False, ref_ids)
    noise, clean = torch.randn(2, gh * gw, 8), torch.randn(1, gh * gw, 8)
    keep = torch.rand(1, gh * gw, 1) < 0.5
    txt = torch.randn(6, num_txt, 12)
    for t_start in (1.0, 0.6):
        # What generate hands in: pure noise at t_start = 1, the mixture below it.
        img = noise if t_start == 1.0 else (1 - t_start) * clean + t_start * noise
        steps = [t_start * t for t in (1.0, 0.6, 0.3, 0.0)]
        out = denoise_torus(
            model,
            img,
            txt,
            geo,
            steps,
            4.0,
            2.0,
            ref=x_ref[:1, gh * gw :],
            keep=keep,
            clean=clean,
            noise=noise,
        )
        kept = keep.expand_as(out)
        assert torch.equal(out[kept], clean.expand_as(out)[kept]), t_start
        assert (out - clean)[~kept].abs().min() > 1e-4, t_start
    print("ok   keep: kept tokens are exactly clean, from t=1 and from a partially noised start")

    # 7
    for num_steps, seq_len in ((4, 48), (50, 1024)):
        assert schedule_from(num_steps, seq_len, 1.0) == get_schedule(num_steps, seq_len)
        for t_start in (0.4, 0.55, 0.7, 0.999):
            s = schedule_from(num_steps, seq_len, t_start)
            assert len(s) == num_steps + 1 and abs(s[0] - t_start) < 1e-6 and s[-1] == 0.0, (t_start, s)
            assert all(a > b for a, b in zip(s, s[1:])), (t_start, s)
    print("ok   schedule_from: starts at t_start, ends at 0, strictly decreasing")

    # 4
    ae = AutoEncoder(AutoEncoderParams(ch=32)).eval()
    z = torch.randn(1, 128, 4, 5)
    for wrap in [(True, True), (False, True), (False, False)]:
        assert decode_torus(ae, z, wrap, pad=2).shape == (1, 3, 64, 80), wrap
    print("ok   decode_torus shapes")
    print("all passed")


if __name__ == "__main__":
    main()
