"""Seconds on a CPU, random toy weights, no checkpoint:  PYTHONPATH=src uv run python scripts/qwen_selftest.py

1. TorusRope in mode "none" returns exactly the stock QwenImage21Rope table (t2i and with a condition image).
2. torus_attention == attention written pair by pair from the displacement rule, for every mode,
   with a prefix of text and condition-image tokens, with and without unanchor_text.
3. Hand-written processor with wrap off == the stock processor, prefill and KV-cached decode alike.
4. With `unanchor_text`, rolling the target latent rolls the output (nearest mode; periodic mode with
   every plane snapped). With the text anchored, or with sub-cycle planes kept, it must not.
5. Periodic mode: the fused path (stock processor + snapped table) == the hand-written one.
6. `schedule` == the stock FlowMatchEulerDiscreteScheduler with the checkpoint's config; t_start variants.
7. `generate` end to end on a fake pipe: text-to-image, with condition images, with an init image and
   a kept region (kept pixels come out exact), and a zero-weight branch is genuinely dead.
8. decode_latents shapes.
"""

import itertools

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    QwenImage21KVCache,
    QwenImage21Rope,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from qwen_torus.attention import TorusAttnProcessor, TorusContext, install, rotate, torus_attention
from qwen_torus.generate import decode_latents, generate, guidance_weights, schedule
from qwen_torus.rope import AXES, PeConfig, TorusRope, build_geometry, inv_freq, token_positions

torch.manual_seed(0)
AXES_DIM = (8, 12, 12)  # head dim 32: 4 + 6 + 6 planes
CTX = 16


def toy_transformer(num_layers=2):
    return QwenImage21Transformer2DModel(
        patch_size=1, in_channels=4, out_channels=4, num_layers=num_layers, attention_head_dim=32,
        num_attention_heads=2, context_in_dim=CTX, mlp_ratio=2, axes_dims_rope=AXES_DIM,
    ).eval()  # fmt: skip


def toy_vae():
    return AutoencoderKLQwenImage21(
        base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1,
        latents_mean=[0.0] * 4, latents_std=[1.0] * 4, in_channels=4, out_channels=4,
    ).eval()  # fmt: skip


def vlm_layout(text: str, cond_grids: list[tuple[int, int]]) -> torch.Tensor:
    """[L] bool mask of a fake Qwen3-VL sequence: text, then each image as (text, slots, text), then text."""
    parts = [torch.zeros(3, dtype=torch.bool)]
    for gh, gw in cond_grids:
        parts += [
            torch.zeros(1, dtype=torch.bool),
            torch.ones(gh * gw // 4, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
        ]
    parts.append(torch.zeros(2 + len(text.split()), dtype=torch.bool))
    return torch.cat(parts)


class FakePipe:
    """What `generate` reads from QwenTorusPipe, with random embeddings in place of the text encoder."""

    def __init__(self):
        self.transformer, self.vae = toy_transformer(), toy_vae()
        self.image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=4)
        self.scheduler_config = {}
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.ctx = TorusContext()
        self.model_name = "toy"

    def encode_prompt(self, text, images):
        grids = [(im.height // 16, im.width // 16) for im in images or []]
        mask = vlm_layout(text, grids)
        g = torch.Generator().manual_seed(abs(hash((text, len(grids)))) % 2**31)
        return torch.randn(1, len(mask), CTX, generator=g), mask[None]

    def free(self):
        pass


def pairwise_target_attention(
    q, k, v, k_prefix, v_prefix, pe_all, positions, n_t, grid, cfg, rope, prefix_is_text
):
    """The definition, with no trick. Target rows only. q, k, v: [B, N_t, H, D] unrotated; k_prefix rotated
    by the stock table; `positions` = (frame, h, w) of every joint token; pe_all the joint rotary table."""
    B, N, H, D = q.shape
    P = k_prefix.shape[1]
    tables = rope.tables(grid)
    frame, hpos, wpos = (p.double() for p in positions)
    qc = torch.view_as_complex(q.double().reshape(B, N, H, -1, 2) * D**-0.5)  # [B, N, H, planes]
    kc_t = torch.view_as_complex(k.double().reshape(B, N, H, -1, 2))
    # prefix keys arrive rotated; undo it to work with the raw key and an explicit displacement
    kc_p = (
        torch.view_as_complex(k_prefix.double().reshape(B, P, H, -1, 2))
        * pe_all[:P].conj().to(torch.complex128)[None, :, None]
    )
    kc = torch.cat([kc_p, kc_t], dim=1)
    edges = [sum(d // 2 for d in AXES_DIM[:i]) for i in range(4)]
    logits = torch.zeros(B, H, N, P + N, dtype=torch.float64)
    is_img = torch.arange(P + N) >= P
    is_text_key = torch.cat([prefix_is_text.bool(), torch.zeros(N, dtype=torch.bool)])
    for a, axis in enumerate(AXES):
        pos = {"f": frame, "h": hpos, "w": wpos}[axis]
        pq, pk = pos[P:], pos
        stock = inv_freq(AXES_DIM[a], rope.theta).double()
        eff = stock if axis == "f" else tables[axis].double()
        # Every token is rotated by its own table at its own coordinate: the target by the effective
        # one, the prefix by the stock one unless scope == "all". The angle of a pair is the difference.
        inv_key = torch.where(
            is_img[:, None], eff[None, :], (eff if cfg.scope == "all" else stock)[None, :]
        )  # [P+N, M]
        angle = inv_key[None] * pk[None, :, None] - eff[None, None, :] * pq[:, None, None]  # [N, P+N, M]
        if axis in ("h", "w"):
            n = grid[0] if axis == "h" else grid[1]
            do_wrap = cfg.wrap_h if axis == "h" else cfg.wrap_w
            if do_wrap and cfg.nearest:  # target-target pairs: the nearest periodic copy of the displacement
                d_near = ((pk[None, :] - pq[:, None]) + n // 2) % n - n // 2
                angle = torch.where(is_img[None, :, None], eff[None, None, :] * d_near[..., None], angle)
            if cfg.unanchor_text:  # text keys: the query stands at coordinate 0
                angle = torch.where(is_text_key[None, :, None], inv_key[None] * pk[None, :, None], angle)
        rot = torch.polar(torch.ones_like(angle), angle)
        s = slice(edges[a], edges[a + 1])
        logits += torch.einsum("bqhm,bkhm,qkm->bhqk", qc[..., s].conj(), kc[..., s], rot).real
    attn = torch.softmax(logits, dim=-1) @ torch.cat([v_prefix, v], dim=1).double().transpose(1, 2)
    return attn.transpose(1, 2)


def roll_tokens(x, gh, shift):
    grid = torch.roll(x.reshape(x.shape[0], gh, -1, x.shape[-1]), shift, (1, 2))
    return grid.reshape(x.shape)


@torch.no_grad()
def main():
    model = toy_transformer()
    stock_rope = QwenImage21Rope(theta=10000, axes_dim=list(AXES_DIM))
    rope = TorusRope(stock_rope)

    # 1. the rope wrapper reproduces the stock table
    for shapes, text in (([(1, 4, 6)], "a b"), ([(1, 2, 4), (1, 4, 6)], "a b c")):
        mask = torch.cat(
            [
                vlm_layout(text, [s[1:] for s in shapes[:-1]]),
                torch.ones(shapes[-1][1] * shapes[-1][2] // 4, dtype=torch.bool),
            ]
        )
        joint = torch.repeat_interleave(mask, torch.where(mask, 4, 1))
        want = stock_rope(shapes, joint, torch.device("cpu"))
        rope.cfg = PeConfig(mode="none")
        got = rope(shapes, joint, torch.device("cpu"))
        err = (got - want).abs().max().item()
        assert err < 1e-5, (shapes, err)
    print("ok   TorusRope(mode=none) == stock QwenImage21Rope")

    # 2. hand-written attention == pairwise definition, with a prefix of text and one condition image
    gh, gw = 4, 6
    shapes = [(1, 2, 4), (1, gh, gw)]
    mask = torch.cat([vlm_layout("x y", [(2, 4)]), torch.ones(gh * gw // 4, dtype=torch.bool)])
    joint = torch.repeat_interleave(mask, torch.where(mask, 4, 1))
    positions = token_positions(shapes, joint)
    n_t, P = gh * gw, len(joint) - gh * gw
    prefix_is_text = ~joint[:P]
    configs = [
        PeConfig(mode="nearest"), PeConfig(mode="nearest", wrap_h=False), PeConfig(mode="none"),
        PeConfig(mode="periodic", min_cycles=0.5), PeConfig(mode="periodic", min_cycles=0.0, low_freq="zero"),
        PeConfig(mode="periodic", scope="all", min_cycles=0.3, rounding="floor"),
        PeConfig(mode="both", min_cycles=0.5, low_freq="fundamental"),
        PeConfig(mode="periodic", min_cycles=0.2, max_rel_error=0.3, rounding="ceil"),
    ]  # fmt: skip
    # the toy grid is 4x6: the fastest plane turns 0.64 / 0.95 times across it, so min_cycles < 1 is
    # what makes snapping happen here at all (on a 64-token side the stock threshold of 1 snaps 8 planes)
    assert (rope.tables((gh, gw))["w"] != inv_freq(AXES_DIM[2], rope.theta)).any() or not rope.cfg.periodic
    for cfg, unanchor in itertools.product(configs, (False, True)):
        cfg = PeConfig.from_dict({**cfg.to_dict(), "unanchor_text": unanchor, "q_chunk": 7})
        geo = build_geometry(rope, cfg, (gh, gw))
        pe_all = rope(shapes, joint, torch.device("cpu"))
        q, k, v = torch.randn(3, 1, n_t, 2, 32).unbind(0)
        kp, vp = torch.randn(2, 1, P, 2, 32).unbind(0)
        kp_rot = rotate(kp, pe_all[:P])
        got = torus_attention(q, k, v, kp_rot, vp, pe_all[P:], geo, prefix_is_text, None, cfg.q_chunk)
        want = pairwise_target_attention(
            q, k, v, kp_rot, vp, pe_all, positions, n_t, (gh, gw), cfg, rope, prefix_is_text
        )
        err = (got.double() - want).abs().max().item()
        assert err < 1e-5, (cfg, err)
    print("ok   torus_attention == pairwise definition for every mode, with text and a condition image")

    # 3. manual processor with wrap off == stock processor, prefill and cached decode
    ctx = TorusContext()
    embeds = torch.randn(
        1, int((~mask[: len(mask) - n_t // 4]).sum()) + int(mask[: len(mask) - n_t // 4].sum()), CTX
    )
    embeds = torch.randn(1, len(mask) - n_t // 4, CTX)
    hidden = torch.randn(1, 2 * 4 + n_t, 4)
    t = torch.tensor([0.7])
    kwargs = dict(
        encoder_hidden_states=embeds, timestep=t, img_shapes=[shapes], img_mask=mask[None], return_dict=False
    )
    model.set_attn_processor(QwenImage21AttnProcessor())
    stock_out = model(hidden_states=hidden, **kwargs)[0]
    cfg = PeConfig(mode="nearest", wrap_h=False, wrap_w=False, q_chunk=5)
    install(model, ctx, cfg)
    assert isinstance(
        model.transformer_blocks[0].attn.processor, QwenImage21AttnProcessor
    ), "no wrap: fused path expected"
    cfg = PeConfig(mode="none", unanchor_text=False, q_chunk=5)
    model.set_attn_processor(TorusAttnProcessor(ctx))  # force the hand-written path
    ctx.geo = build_geometry(model.pos_embed, cfg, (gh, gw))
    ctx.prefix_is_text = prefix_is_text
    cache = QwenImage21KVCache(len(model.transformer_blocks))
    out = model(hidden_states=hidden, kv_cache=cache, kv_cache_mode="extract", **kwargs)[0]
    err = (out - stock_out).abs().max().item()
    assert err < 1e-4, err
    stock_cache = QwenImage21KVCache(len(model.transformer_blocks))
    model.set_attn_processor(QwenImage21AttnProcessor())
    model(hidden_states=hidden, kv_cache=stock_cache, kv_cache_mode="extract", **kwargs)
    # decode still takes the full hidden states; the transformer drops the prefix rows itself
    stock_dec = model(hidden_states=hidden, kv_cache=stock_cache, kv_cache_mode="cached", **kwargs)[0]
    model.set_attn_processor(TorusAttnProcessor(ctx))
    dec = model(hidden_states=hidden, kv_cache=cache, kv_cache_mode="cached", **kwargs)[0]
    err = (dec - stock_dec).abs().max().item()
    assert err < 1e-4, err
    assert (
        dec - out[:, -n_t:]
    ).abs().max().item() < 1e-4, "cached decode must equal the prefill's target rows"
    print(
        f"ok   hand-written processor, wrap off: equals stock forward, prefill and KV-cached decode ({err:.1e})"
    )

    # 4. roll equivariance (text-to-image: no condition image, since a reference pins the picture)
    shapes_t2i = [(1, gh, gw)]
    mask_t2i = torch.cat([vlm_layout("p q r", []), torch.ones(n_t // 4, dtype=torch.bool)])
    embeds_t2i = torch.randn(1, len(mask_t2i) - n_t // 4, CTX)
    kw = dict(
        encoder_hidden_states=embeds_t2i,
        timestep=t,
        img_shapes=[shapes_t2i],
        img_mask=mask_t2i[None],
        return_dict=False,
    )
    x = torch.randn(1, n_t, 4)
    cases = [
        (PeConfig(mode="nearest", unanchor_text=True), True),
        (PeConfig(mode="nearest", unanchor_text=False), False),
        (PeConfig(mode="periodic", min_cycles=0.0, low_freq="zero", unanchor_text=True), True),
        (PeConfig(mode="periodic", min_cycles=0.0, low_freq="fundamental", unanchor_text=True), True),
        (PeConfig(mode="periodic", min_cycles=1.0, low_freq="keep", unanchor_text=True), False),
        (PeConfig(mode="both", min_cycles=1.0, low_freq="keep", unanchor_text=True), True),
    ]
    for cfg, expect in cases:
        cfg.q_chunk = 5
        install(model, ctx, cfg)
        ctx.geo = build_geometry(model.pos_embed, cfg, (gh, gw))
        ctx.prefix_is_text = torch.ones(len(mask_t2i) - n_t // 4, dtype=torch.bool)
        out = model(hidden_states=x, **kw)[0][:, -n_t:]
        worst = 0.0
        for shift in [(1, 0), (0, 1), (gh // 2, gw // 2), (gh - 1, 3)]:
            rolled = model(hidden_states=roll_tokens(x, gh, shift), **kw)[0][:, -n_t:]
            worst = max(worst, (rolled - roll_tokens(out, gh, shift)).abs().max().item())
        if expect:
            assert worst < 1e-4, (cfg, worst)
        else:
            assert worst > 1e-3, (cfg, worst)
        print(f"ok   {cfg.mode:8s} min_cycles={cfg.min_cycles} low_freq={cfg.low_freq} unanchor={cfg.unanchor_text}: "
              f"{'roll-equivariant' if expect else 'NOT roll-equivariant, as it should be'} ({worst:.1e})")  # fmt: skip

    # 5. periodic: fused (stock processor on the snapped table) == hand-written
    cfg = PeConfig(mode="periodic", min_cycles=0.5)
    install(model, ctx, cfg)
    assert isinstance(model.transformer_blocks[0].attn.processor, QwenImage21AttnProcessor)
    fused = model(hidden_states=hidden, **kwargs)[0]
    ctx.geo = build_geometry(model.pos_embed, cfg, (gh, gw))
    ctx.prefix_is_text = prefix_is_text
    model.set_attn_processor(TorusAttnProcessor(ctx))
    manual = model(hidden_states=hidden, **kwargs)[0]
    err = (fused - manual).abs().max().item()
    assert err < 1e-4, err
    print(f"ok   periodic mode: fused path == hand-written path ({err:.1e})")

    # 6. schedule
    config = dict(base_image_seq_len=256, max_image_seq_len=8192, base_shift=0.5, max_shift=0.9, shift_terminal=0.02,
                  num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=True, time_shift_type="exponential")  # fmt: skip
    sched = FlowMatchEulerDiscreteScheduler(**config)
    for num_steps, seq_len in ((4, 48), (50, 4096), (40, 16384)):
        mu = 0.5 + (0.9 - 0.5) / (8192 - 256) * (seq_len - 256)
        sched.set_timesteps(num_steps, sigmas=np.linspace(1.0, 1 / num_steps, num_steps), mu=mu)
        want = sched.sigmas.tolist()
        got = schedule(num_steps, seq_len, 1.0, config)
        assert len(got) == len(want) and max(abs(a - b) for a, b in zip(got, want)) < 1e-5, (got, want)
        for t_start in (0.4, 0.6, 0.999):
            s = schedule(num_steps, seq_len, t_start, config)
            assert (
                len(s) == num_steps + 1
                and abs(s[0] - t_start) < 1e-9
                and s[-1] == 0.0
                and abs(s[-2] - 0.02) < 1e-9
            )
            assert all(a > b for a, b in zip(s, s[1:])), s
    print(
        "ok   schedule: stock sigmas at t_start = 1; starts at t_start, ends 0.02 -> 0, strictly decreasing"
    )

    # 7. generate, end to end
    pipe = FakePipe()
    W, H = 96, 64
    tile = generate(pipe, "a tile", seed=1, width=W, height=H, num_steps=3, pe=PeConfig(q_chunk=5))
    assert tile.shape == (H, W, 3) and tile.dtype == torch.uint8
    # the transformer comes back stock: fused processor, no wrapping in the rope wrapper, no geometry
    assert isinstance(pipe.transformer.transformer_blocks[0].attn.processor, QwenImage21AttnProcessor)
    assert not pipe.transformer.pos_embed.cfg.periodic and pipe.ctx.geo is None
    # seamless off == stock position encoding: the wrapper's table is the stock module's, and the
    # processor it installs is the stock one, even with mode "nearest" selected
    from qwen_torus.generate import STOCK

    install(pipe.transformer, pipe.ctx, PeConfig(mode="nearest", wrap_h=False, wrap_w=False))
    assert isinstance(pipe.transformer.transformer_blocks[0].attn.processor, QwenImage21AttnProcessor)
    assert not pipe.transformer.pos_embed.cfg.nearest and STOCK.needs_manual_attention is False
    ref = Image.fromarray(np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8))
    style = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 70, 3), dtype=np.uint8))
    for cfg in (
        PeConfig(mode="nearest", unanchor_text=True, q_chunk=5),
        PeConfig(mode="periodic"),
        PeConfig(mode="none"),
    ):
        tile = generate(pipe, "fill it", seed=1, width=W, height=H, num_steps=3, guidance=3.0, geo_guidance=5.0,
                        pe=cfg, cond_images=[ref, style], ref_max_pixels=64 * 64)  # fmt: skip
        assert tile.shape == (H, W, 3)
    keep_mask = Image.new("RGB", (W, H), "black")
    keep_mask.paste(Image.new("RGB", (32, 32), "white"), (32, 16))
    out = generate(pipe, "keep the middle", seed=2, width=W, height=H, num_steps=3, init_image=ref, keep_mask=keep_mask,
                   t_start=0.6, pe=PeConfig(q_chunk=5))  # fmt: skip
    kept = np.asarray(ref)[16:48, 32:64]
    assert np.array_equal(out[16:48, 32:64].numpy(), kept), "kept pixels must come back exactly"
    assert not np.array_equal(out.numpy(), np.asarray(ref)), "the free region must change"
    # a dead branch: with the same embeddings for uncond and cond, guidance = 1 and guidance = 4 agree
    pipe.encode_prompt = (lambda f: lambda text, images: f("same", images))(pipe.encode_prompt)
    a = generate(
        pipe,
        "p",
        seed=3,
        width=W,
        height=H,
        num_steps=2,
        guidance=4.0,
        geo_guidance=0.0,
        pe=PeConfig(q_chunk=5),
    )
    b = generate(
        pipe,
        "p",
        seed=3,
        width=W,
        height=H,
        num_steps=2,
        guidance=1.0,
        geo_guidance=0.0,
        pe=PeConfig(q_chunk=5),
    )
    assert (a.int() - b.int()).abs().max() <= 1, "dropping a zero-weight branch changed the trajectory"
    assert guidance_weights(1.0, 2.0) == [0.0, -1.0, 2.0] and guidance_weights(1.0, 0.0) == [0.0, 1.0, 0.0]
    print("ok   generate: t2i, condition images, init + keep (kept pixels exact), dead branch dropped")

    # 8. decode shapes
    x = torch.randn(1, 4 * 6, 4)
    for wrap in [(True, True), (False, True), (False, False)]:
        assert decode_latents(pipe, x, (4, 6), wrap, pad=2).shape == (1, 4, 64, 96), wrap
    print("ok   decode_latents shapes")

    # 9. seam_ratio: ~1 on a field that repeats, well above 1 on one with an edge; generate reports it
    from qwen_torus.prep import seam_ratio

    # The ratio compares one pair of rows with the average pair, so it is a statistical measure: a
    # textured field that wraps (circularly low-passed noise -- what the FFT makes periodic) scores
    # ~1, the same field plus a ramp scores far above it. A single smooth sinusoid would not: its
    # wrap pair sits at one phase of the cycle, where the neighbour difference is not the average one.
    rng = np.random.default_rng(0)
    spectrum = np.fft.fft2(rng.normal(size=(96, 128, 3)), axes=(0, 1))
    fy, fx = np.meshgrid(np.fft.fftfreq(96), np.fft.fftfreq(128), indexing="ij")
    spectrum *= (np.hypot(fy, fx) < 0.3)[..., None]
    periodic = np.fft.ifft2(spectrum, axes=(0, 1)).real
    yy, xx = np.meshgrid(np.arange(96), np.arange(128), indexing="ij")
    edged = periodic + np.stack([xx / 128.0, yy / 96.0, xx * 0], -1) * periodic.std() * 20
    assert all(abs(r - 1) < 0.25 for r in seam_ratio(periodic)), seam_ratio(periodic)
    assert all(r > 5 for r in seam_ratio(edged)), seam_ratio(edged)
    info = {}
    generate(pipe, "p", seed=3, width=W, height=H, num_steps=2, pe=PeConfig(q_chunk=5), info=info)
    assert set(info) == {"seam_latent", "seam_pixels"} and len(info["seam_latent"]) == 2
    print(
        f"ok   seam_ratio: periodic {seam_ratio(periodic)[1]:.2f}, edged {seam_ratio(edged)[1]:.1f}; generate reports it"
    )
    print("all passed")


if __name__ == "__main__":
    main()
