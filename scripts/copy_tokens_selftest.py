"""Seconds on a CPU, random toy weights, no checkpoint:  uv run python scripts/copy_tokens_selftest.py

1. The 2 x 2 example of the module docstring, band 1: 16 keys on {-1, 0, 1, 2}^2, the copy at (i, j)
   being the token at (i mod 2, j mod 2).
2. copy_attention == attention written out pair by pair from the displacement, with the padded key
   grid built token by token in Python loops; text and reference queries never see a copy.
3. band 0: copy_forward == the stock Flux2.forward, with and without reference tokens.
4. The copies reach the output: band 1 != band 0 for the image tokens, and the text and reference
   outputs of one attention layer are the same for every band.
5. seamless.copy_tokens.KleinCopy end to end on toy weights: picture in, picture of the same size
   out, deterministic, and band 0 / band 2 / the RoPE control are three different pictures.
"""

import torch
from einops import rearrange
from PIL import Image

from flux2.copy_tokens import build_copy_geometry, copy_attention, copy_forward
from flux2.model import Flux2, rope
from seamless.copy_tokens import KleinCopy
from torus_selftest import ToyParams, make_ids, make_ref_ids


def padded_attention(q, k, v, ids, num_txt, grid, band, theta=2000):
    """The definition, with no trick: for every image query, a key per token of the padded grid, and
    logit[p, key] = sum over axes of x_p^T R(key position - p) x_key."""
    gh, gw = grid
    n = ids.shape[1]
    src, pos = list(range(n)), [ids[0, i].tolist() for i in range(n)]
    for i in range(-band, gh + band):
        for j in range(-band, gw + band):
            if 0 <= i < gh and 0 <= j < gw:
                continue
            src.append(num_txt + (i % gh) * gw + (j % gw))  # make_ids lists the grid row-major
            pos.append([0, i, j, 0])
    pos = torch.tensor(pos)  # [n + e, 4]
    k, v = k[:, :, src], v[:, :, src]
    logits = 0
    for axis in range(4):
        d = pos[None, :, axis] - ids[0, :, axis, None]  # [query, key]
        rot = rope(d, 32, theta).double()
        qa = q[..., 32 * axis : 32 * axis + 32].reshape(*q.shape[:-1], 16, 2).double()
        ka = k[..., 32 * axis : 32 * axis + 32].reshape(*k.shape[:-1], 16, 2).double()
        logits = logits + torch.einsum("bhpmi,pqmij,bhqmj->bhpq", qa, rot, ka)
    is_img = (torch.arange(n) >= num_txt) & (torch.arange(n) < num_txt + gh * gw)
    logits[:, :, ~is_img, n:] = -torch.inf  # text and reference see the stock keys only
    attn = torch.softmax(logits * q.shape[-1] ** -0.5, dim=-1) @ v.double()
    return rearrange(attn, "b h n d -> b n (h d)")


@torch.no_grad()
def main():
    torch.manual_seed(0)
    model = Flux2(ToyParams()).eval()
    num_txt = 5

    # 1
    x_ids, ctx_ids = make_ids(2, 2, num_txt)
    geo = build_copy_geometry(model, x_ids, ctx_ids, (2, 2), band=1)
    copies = geo.key_ids[num_txt + 4 :]
    assert copies.shape == (12, 4), copies.shape  # 16 keys, 4 of them the image itself
    assert set(map(tuple, copies[:, 1:3].tolist())) == {(i, j) for i in (-1, 0, 1, 2) for j in (-1, 0, 1, 2)} - {
        (i, j) for i in (0, 1) for j in (0, 1)
    }
    for (_, i, j, _), src in zip(copies.tolist(), geo.key_src[num_txt + 4 :].tolist()):
        assert geo.key_ids[src, 1:3].tolist() == [i % 2, j % 2], (i, j, src)
    print("ok   2x2 image, band 1: 16 keys on {-1,0,1,2}^2, copy (i, j) is token (i mod 2, j mod 2)")

    # 2, 3, 4
    gh, gw = 6, 8
    x_ids, ctx_ids = make_ids(gh, gw, num_txt)
    ref_ids = make_ref_ids(gh, gw)
    for with_ref in (False, True):
        ids = torch.cat((ctx_ids, x_ids, ref_ids) if with_ref else (ctx_ids, x_ids), 1)
        n = ids.shape[1]
        q, k, v = torch.randn(3, 2, 2, n, 128).unbind(0)
        stock = None
        for band in (0, 1, 3, 6):
            geo = build_copy_geometry(model, x_ids, ctx_ids, (gh, gw), band, ref_ids if with_ref else None)
            got = copy_attention(q, k, v, geo)
            want = padded_attention(q, k, v, ids, num_txt, (gh, gw), band)
            err = (got - want).abs().max().item()
            assert err < 1e-5, (with_ref, band, err)
            if band == 0:
                stock = got
            else:
                assert (got[:, num_txt : num_txt + gh * gw] - stock[:, num_txt : num_txt + gh * gw]).abs().max() > 1e-3
                assert torch.allclose(got[:, :num_txt], stock[:, :num_txt], atol=1e-5)
                assert torch.allclose(got[:, num_txt + gh * gw :], stock[:, num_txt + gh * gw :], atol=1e-5)
        print(f"ok   {'[txt, img, ref]' if with_ref else '[txt, img]'}: pairwise attention at bands 0, 1, 3, 6; "
              "copies move the image and leave text and ref alone")

        x = torch.randn(2, n - num_txt, 8)
        ctx = torch.randn(2, num_txt, 12)
        t = torch.tensor([0.7, 0.3])
        geo = build_copy_geometry(model, x_ids, ctx_ids, (gh, gw), 0, ref_ids if with_ref else None)
        out = copy_forward(model, x, t, ctx, None, geo)
        stock = model(x, ids[:, num_txt:].expand(2, -1, -1), t, ctx, ctx_ids.expand(2, -1, -1), None)
        err = (out - stock).abs().max().item()
        assert err < 1e-4, (with_ref, err)
        geo = build_copy_geometry(model, x_ids, ctx_ids, (gh, gw), 1, ref_ids if with_ref else None)
        assert (copy_forward(model, x, t, ctx, None, geo) - stock).abs().max() > 1e-3, "band 1 changed nothing"
        print(f"ok   band 0 equals stock forward ({err:.1e}), band 1 does not")

    # 5
    picture = Image.fromarray(torch.randint(0, 256, (160, 192, 3), dtype=torch.uint8).numpy())
    klein = KleinCopy.toy(["fill", "moss"], num_steps=2)
    out = klein("fill", seed=0, band=2, reference=picture)
    assert out.size == picture.size and out.mode == "RGB", out.size
    assert out.tobytes() == klein("fill", seed=0, band=2, reference=picture).tobytes(), "same seed, different picture"
    assert out.tobytes() != klein("fill", seed=0, band=0, reference=picture).tobytes(), "the band changed nothing"
    assert out.tobytes() != klein("fill", seed=0, band=None, reference=picture).tobytes(), "band 2 equals the RoPE control"
    t2i = klein("moss", seed=0, band=1, size=160)  # 10 tokens: decode_torus continues the latent 9 past each edge
    assert t2i.size == (160, 160) and t2i.tobytes() != klein("moss", seed=0, band=None, size=160).tobytes()
    print("ok   KleinCopy: inpaint and t2i, picture out; deterministic; band 0, band 2 and rope differ")
    print("all passed")


if __name__ == "__main__":
    main()
