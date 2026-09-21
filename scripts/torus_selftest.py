"""Seconds on a CPU, random toy weights, no checkpoint:  PYTHONPATH=src python scripts/torus_selftest.py

1. torus_attention == attention written pair by pair from the nearest-copy displacement.
2. wrap off: torus_forward == the stock Flux2.forward.
3. unanchor_text: rolling the input latent on the torus rolls the output, i.e. the network cannot
   tell where the image was cut. With the text anchored (the default) the same test must fail.
4. decode_torus returns the right size.
"""

import itertools
from dataclasses import dataclass, field

import torch
from einops import rearrange

from flux2.autoencoder import AutoEncoder, AutoEncoderParams
from flux2.model import Flux2, rope
from flux2.torus import build_torus_geometry, decode_torus, torus_attention, torus_forward


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


def roll(tokens, gh, shift):
    """Cyclic shift of a [B, gh * gw, C] token sequence on its grid."""
    grid = torch.roll(rearrange(tokens, "b (h w) c -> b h w c", h=gh), shift, (1, 2))
    return rearrange(grid, "b h w c -> b (h w) c")


def pairwise_attention(q, k, v, ids, num_txt, grid, wrap, unanchor_text, theta=2000):
    """The definition, with no trick: logit[p, q] = sum over axes of x_p^T R(displacement) x_q."""
    n_tok = ids.shape[1]
    is_img = torch.arange(n_tok) >= num_txt
    img_img = is_img[:, None] & is_img[None, :]
    logits = 0
    for axis in range(4):
        pos = ids[0, :, axis]
        d = pos[None, :] - pos[:, None]
        if axis in (1, 2):
            n = grid[axis - 1]
            if wrap[axis - 1]:
                d = torch.where(img_img, (d + n // 2) % n - n // 2, d)
            if unanchor_text:
                d = torch.where(img_img, d, 0)
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

    # 4
    ae = AutoEncoder(AutoEncoderParams(ch=32)).eval()
    z = torch.randn(1, 128, 4, 5)
    for wrap in [(True, True), (False, True), (False, False)]:
        assert decode_torus(ae, z, wrap, pad=2).shape == (1, 3, 64, 80), wrap
    print("ok   decode_torus shapes")
    print("all passed")


if __name__ == "__main__":
    main()
