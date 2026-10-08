"""Seconds on a CPU, random toy weights, no checkpoint:  uv run python scripts/torus_selftest.py

Every test that involves the geometry runs for both ways of making the RoPE periodic, the nearest
copy of the displacement and the quantized frequencies (`rope`).

0. The quantized frequencies at 64 tokens are what torus.py says they are, and the per-plane rules
   do what they say: fundamental never reaches zero, keep leaves the trained frequency, an integer
   sets the cycle count. Kept planes are not periodic, so with them the roll test of 3 must fail
   even with the text unanchored.
1. torus_attention == attention written pair by pair from its definition: the nearest-copy
   displacement with the stock frequencies, or the raw displacement with the quantized ones.
2. wrap off: torus_forward == the stock Flux2.forward.
3. unanchor_text: rolling the input latent on the torus rolls the output, i.e. the network cannot
   tell where the image was cut. With the text anchored (the default) the same test must fail.
4. decode_torus returns the right size.
5. With reference tokens ([txt, img, ref], ref grid larger than the image grid): 1 and 2 again.
6. denoise_torus: wrap off and guidance 1 is the stock sampler, `sampling.denoise`; guidance != 1
   is classifier-free guidance, which with the empty prompt set equal to the prompt is guidance 1.
7. seamless.klein.Klein end to end on toy weights: a picture in, a picture of the same size out,
   the same for the same seed, different with the wrap off, the other rope or another seed.
"""

import itertools
from dataclasses import dataclass, field

import torch
from einops import rearrange
from PIL import Image

from flux2.autoencoder import AutoEncoder, AutoEncoderParams
from flux2.model import Flux2
from flux2.sampling import denoise
from flux2.torus import (
    build_torus_geometry,
    decode_torus,
    denoise_torus,
    frequencies,
    quantize,
    rotations,
    torus_attention,
    torus_forward,
)
from seamless.klein import Klein


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


def pairwise_attention(q, k, v, ids, num_txt, grid, wrap, unanchor_text, rope="nearest", theta=2000):
    """The definition, with no trick: logit[p, q] = sum over axes of x_p^T R(displacement) x_q.
    Tokens are [txt, img, ref]; the img tokens are the grid[0] * grid[1] right after the text.
    "nearest" wraps the img-img displacement, "quantized" wraps the frequencies of every pair."""
    n_tok = ids.shape[1]
    index = torch.arange(n_tok)
    is_img = (index >= num_txt) & (index < num_txt + grid[0] * grid[1])
    img_img = is_img[:, None] & is_img[None, :]
    txt_pair = (index < num_txt)[:, None] | (index < num_txt)[None, :]
    logits = 0
    for axis in range(4):
        pos = ids[0, :, axis]
        d = pos[None, :] - pos[:, None]
        omega = frequencies(32, theta)
        if axis in (1, 2):
            n = grid[axis - 1]
            if wrap[axis - 1] and rope == "quantized":
                omega = quantize(omega, n)
            elif wrap[axis - 1]:
                d = torch.where(img_img, (d + n // 2) % n - n // 2, d)
            if unanchor_text:
                d = torch.where(txt_pair, 0, d)
        rot = rotations(d[..., None].float() * omega).double()  # [N, N, 16, 2, 2]
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
    ropes = ["nearest", "quantized"]

    # 0
    omega = quantize(frequencies(32, 2000), 64)
    assert torch.equal(omega * 64 / (2 * torch.pi), (omega * 64 / (2 * torch.pi)).round()), "not periodic"
    assert (omega == 0).sum() == 9 and len(set(omega[omega > 0].tolist())) == 5, omega
    trained = frequencies(32, 2000)
    fundamental = quantize(trained, 64, ["fundamental"] * 16)
    assert torch.equal(fundamental[:7], omega[:7]) and (fundamental[7:] == 2 * torch.pi / 64).all()
    mixed = quantize(trained, 64, ["round"] * 7 + ["keep"] * 8 + [3])
    assert torch.equal(mixed[:7], omega[:7]) and torch.allclose(mixed[7:15], trained[7:15]), mixed
    assert torch.isclose(mixed[15], torch.tensor(3 * 2 * torch.pi / 64)), mixed[15]
    print("ok   quantized frequencies at n = 64: 9 planes blind to position, 7 left on 5 frequencies; the rules")

    for (gh, gw), wrap, unanchor_text, rope in itertools.product(
        [(6, 8), (5, 7), (2, 3)], [(True, True), (False, True), (False, False)], [False, True], ropes
    ):
        x_ids, ctx_ids = make_ids(gh, gw, num_txt)
        geo = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), wrap, unanchor_text, rope=rope)
        tag = f"grid {gh}x{gw} wrap {wrap} unanchor_text {unanchor_text} rope {rope}"

        # 1
        q, k, v = torch.randn(3, 2, 2, num_txt + gh * gw, 128).unbind(0)
        got = torus_attention(q, k, v, geo, q_chunk=7)
        want = pairwise_attention(
            q, k, v, torch.cat((ctx_ids, x_ids), 1), num_txt, (gh, gw), wrap, unanchor_text, rope
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
            # On a grid this small every quantized frequency rounds to zero: the image is blind to
            # position, so nothing is left for the text to anchor.
            blind = rope == "quantized" and not any(quantize(frequencies(32, 2000), n).any() for n in (gh, gw))
            if unanchor_text or blind:
                assert worst < 1e-4, (tag, worst)
                print(f"ok   {tag}: roll-equivariant ({worst:.1e}){'  (all frequencies zero)' if blind else ''}")
            else:
                assert worst > 1e-5, (tag, worst)
                print(f"ok   {tag}: NOT roll-equivariant, the text marks the origin ({worst:.1e})")

    # 3, with rules: kept planes are not periodic, fundamental ones are
    gh, gw = 6, 8
    x_ids, ctx_ids = make_ids(gh, gw, num_txt)
    x, ctx, t = torch.randn(2, gh * gw, 8), torch.randn(2, num_txt, 12), torch.tensor([0.7, 0.3])
    for rules, periodic in ((["fundamental"] * 16, True), (["round"] * 8 + ["keep"] * 8, False)):
        geo = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), unanchor_text=True, rope="quantized", rules=rules)
        out = torus_forward(model, x, t, ctx, None, geo)
        rolled = torus_forward(model, roll(x, gh, (2, 3)), t, ctx, None, geo)
        worst = (rolled - roll(out, gh, (2, 3))).abs().max().item()
        assert (worst < 1e-4) == periodic, (rules, worst)
    print("ok   rules: every plane on the fundamental or above is roll-equivariant, a kept plane is not")

    # 5
    x_ids, ctx_ids = make_ids(gh, gw, num_txt)
    ref_ids = make_ref_ids(7, 11)
    n_ref = ref_ids.shape[1]
    x_ref = torch.randn(2, gh * gw + n_ref, 8)
    for wrap, unanchor_text, rope in itertools.product([(True, True), (False, False)], [False, True], ropes):
        geo = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), wrap, unanchor_text, ref_ids, rope)
        q, k, v = torch.randn(3, 2, 2, num_txt + gh * gw + n_ref, 128).unbind(0)
        ids = torch.cat((ctx_ids, x_ids, ref_ids), 1)
        want = pairwise_attention(q, k, v, ids, num_txt, (gh, gw), wrap, unanchor_text, rope)
        err = (torus_attention(q, k, v, geo, q_chunk=7) - want).abs().max().item()
        assert err < 1e-5, ("ref", wrap, unanchor_text, rope, err)
        if wrap == (False, False) and not unanchor_text:
            out = torus_forward(model, x_ref, t, ctx, None, geo)
            img_ids = torch.cat((x_ids, ref_ids), 1).expand(2, -1, -1)
            stock = model(x_ref, img_ids, t, ctx, ctx_ids.expand(2, -1, -1), None)
            err = (out - stock).abs().max().item()
            assert err < 1e-4, ("ref stock", err)
    print("ok   reference tokens: pairwise attention, and wrap off equals stock forward with refs")

    # 6
    noise, txt = torch.randn(2, gh * gw, 8), torch.randn(2, num_txt, 12)
    ref, steps = x_ref[:1, gh * gw :], [1.0, 0.6, 0.3, 0.0]
    flat = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), (False, False), False, ref_ids)
    stock = denoise(
        model,
        noise,
        x_ids.expand(2, -1, -1),
        txt,
        ctx_ids.expand(2, -1, -1),
        steps,
        1.0,
        img_cond_seq=ref.expand(2, -1, -1),
        img_cond_seq_ids=ref_ids.expand(2, -1, -1),
    )
    err = (denoise_torus(model, noise, txt, flat, steps, ref=ref) - stock).abs().max().item()
    assert err < 1e-4, ("stock sampler", err)
    geo = build_torus_geometry(model, x_ids, ctx_ids, (gh, gw), (True, True), False, ref_ids)
    one = denoise_torus(model, noise, txt, geo, steps, ref=ref)
    two = denoise_torus(model, noise, torch.cat((txt, txt)), geo, steps, guidance=3.0, ref=ref)
    err = (one - two).abs().max().item()
    assert err < 1e-4, ("cfg", err)
    assert (one - stock).abs().max() > 1e-4, "the wrap changed nothing"
    print("ok   denoise_torus: wrap off is the stock sampler, CFG with equal branches is guidance 1")

    # 4
    ae = AutoEncoder(AutoEncoderParams(ch=32)).eval()
    z = torch.randn(1, 128, 4, 5)
    for wrap in [(True, True), (False, True), (False, False)]:
        assert decode_torus(ae, z, wrap, pad=2).shape == (1, 3, 64, 80), wrap
    print("ok   decode_torus shapes")

    # 7
    picture = Image.fromarray(torch.randint(0, 256, (160, 192, 3), dtype=torch.uint8).numpy())
    klein = Klein.toy(["fill"], num_steps=2)
    out = klein(picture, "fill", seed=0)
    assert out.size == picture.size and out.mode == "RGB", out.size
    assert out.tobytes() == klein(picture, "fill", seed=0).tobytes(), "same seed, different picture"
    assert out.tobytes() != klein(picture, "fill", seed=1).tobytes(), "the seed changed nothing"
    assert out.tobytes() != Klein.toy(["fill"], num_steps=2, wrap=False)(picture, "fill").tobytes()
    assert out.tobytes() != Klein.toy(["fill"], num_steps=2, rope="quantized")(picture, "fill").tobytes()
    assert (
        out.tobytes() != Klein.toy(["fill"], "You are an inpainter.", num_steps=2)(picture, "fill").tobytes()
    )
    assert Klein.toy(["fill"], guidance=2.0, num_steps=2)(picture, "fill").size == picture.size
    print(
        "ok   Klein: picture in, picture out; deterministic; seed, wrap, rope, system prompt, guidance reach the sampler"
    )
    print("all passed")


if __name__ == "__main__":
    main()
