"""Four-way seamless images from FLUX.2, training free: RoPE with the nearest periodic copy.

An image that tiles -- paste copies of it on a grid and it is continuous everywhere -- is an image
on a torus: column gw-1 is the left neighbour of column 0, row gh-1 is the top neighbour of row 0.

RoPE makes the logit between a query at p and a key at q

    (R(p) x_p)^T (R(q) x_q) = x_p^T R(q - p) x_q,        R(d) = blockdiag(rot(omega_m * d))

so all that position contributes is the displacement d = q - p. There are two ways to make that
periodic. One is to move every omega_m onto a multiple of 2 pi / n, so that R(d + n) = R(d)
(reference_code/position.py, and the high-frequency path of reference_code/erp_utils.py); that
changes the frequencies the model was trained with. The other, this file, keeps every frequency and
changes the displacement: on a circle of n tokens the displacement from p to q is not q - p but its
nearest periodic copy (the "minimum image convention" of periodic-boundary simulations)

    d_near = ((q - p + n/2) mod n) - n/2        in [-n/2, n/2)

Every query sees itself in the middle of the image, with n/2 tokens to either side. No |d_near|
exceeds n/2, so every rotation used here is one the model met in training.

Three facts keep it cheap.

1. d_near is one of d - n, d, d + n, so R(d_near) = R(p)^T R(q + s n) with s in {-1, 0, +1}:
   queries are rotated exactly as always, only the keys come in shifted copies.
2. Two copies per key are enough, not three. A key in the lower half (2q < n) is only ever wanted
   at q or q + n, a key in the upper half only at q or q - n. `build_torus_geometry` asserts it.
3. R is block diagonal over the four axes (t, h, w, l), so a logit is a sum of four partial logits,
   one per axis, and wrapping h only touches the h part, wrapping w only the w part. Two dimensions
   therefore cost 2 + 2 partial products and not 2 x 2 (let alone 3 x 3) full ones: six matmuls
   over 32 dims where stock attention does one over 128, i.e. 1.5x the QK^T work.

What is lost is the fused kernel. Which copy of a key is used depends on the query, so R(d_near) is
no longer R(p)^T R(q') for any single q', the [N, N] logits have to be written out, and
F.scaled_dot_product_attention cannot be used. `torus_attention` does it by hand, in query chunks.

Two choices that are not forced, so they are flagged here rather than buried:

- The tie. For even n the displacement n/2 is as far to the left as to the right. The interval is
  taken half open, [-n/2, n/2), so that d_near is a function of (q - p) mod n alone; that is what
  makes the network exactly equivariant to cyclic shifts. Keeping the raw +-n/2 instead would be
  antisymmetric in (p, q) but would change under a shift.
- The text. FLUX.2 pins every text token at (h=0, w=0), so an image token at (h, w) sees the text at
  displacement (-h, -w): through the text, every image token knows its absolute position and the
  torus has a marked origin. By default that is left alone (least change to what the model saw in
  training). `unanchor_text=True` gives text<->image pairs zero displacement on h and w -- every
  image token sees the text the way the token at (0, 0) always did -- and then nothing in the
  network can tell where the image was cut: roll the input latent and the output rolls with it, to
  floating point (scripts/torus_selftest.py checks this).

Two ways to bring an existing picture in, both optional and independent:

- Reference image (FLUX.2's own image-to-image): its tokens are appended after the image tokens,
  [txt, img, ref], exactly as the stock `denoise` does. A reference is an ordinary flat picture, so
  every pair that involves a ref token keeps its stock displacement; only img-img pairs wrap.
- Untouched region (`keep` in `denoise_torus`): not a condition of the network at all, a constraint
  on the sampler. After every Euler step the kept tokens are overwritten by the clean latent noised
  to the current time, so the free tokens are always denoised next to a correctly-noised version of
  what must stay, across the torus seam as well. Keep the middle of any picture, leave a band at the
  edges free, and the model has to invent the band that makes the picture tile.

The second of those has a failure mode that is worth stating plainly, because the fix is a number
the caller picks. At t = 1 the kept region *is* pure noise -- (1 - t) clean + t noise with t = 1
carries nothing of `clean` -- so the first Euler step decides the free region while knowing nothing
about the picture it has to join. That first velocity is a full-size commitment to a layout drawn
from the prompt alone; later steps can only refine it, so the free region ends up composed for a
picture that is not there. Starting the sampler part way down the trajectory instead, at
timesteps[0] = t_start in roughly 0.4-0.7 with x_t = (1 - t_start) clean + t_start noise over the
*whole* grid, means every step sees the picture. `torus_generate.generate` takes `t_start` and does
this; the price is that the free region begins as a noised copy of what was there, so the lower
t_start goes the less the model will move it.
"""

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from .model import Flux2, apply_rope, timestep_embedding

# Appended to the prompt for the third guidance branch, the analogue of ERP_PROMPT in
# reference_code/erp_utils.py. Untuned.
TORUS_PROMPT = (
    "The whole image is one single seamless tile: its left edge continues into its right edge and "
    "its top edge continues into its bottom edge, so copies of it placed side by side join with no "
    "visible seam. The image is exactly one cell, not a grid or mosaic of smaller repeated tiles, "
    "and no motif or cluster inside it is visibly repeated: irregular organic layout, varied scale, "
    "elements cross the edges. No border, no frame, no vignette."
)

# One-sided receptive field of the AE decoder is ~18 latent pixels (same number as
# decoder_circular_pad_width in reference_code/erp_utils.py), and a token is 2x2 latent pixels.
DECODE_PAD_TOKENS = 9


@dataclass
class TorusGeometry:
    """Everything position contributes, for the joint sequence [txt, img]. Built once per image."""

    pe: Tensor  # [1, 1, N, 64, 2, 2] stock rotations, every token at its own position
    pe_copy: Tensor  # the same, with image tokens moved to their other periodic copy (h and w both)
    use_copy_h: Tensor  # [N, N] bool, [query, key]: on the h axis this pair uses the key's copy
    use_copy_w: Tensor
    txt_pair: Tensor  # [N, N] bool: at least one of the two tokens is text
    dims: dict[str, slice]  # which head dims each axis rotates: t, h, w, l
    num_txt: int
    unanchor_text: bool


def build_torus_geometry(
    model: Flux2,
    x_ids: Tensor,
    ctx_ids: Tensor,
    grid: tuple[int, int],
    wrap: tuple[bool, bool] = (True, True),
    unanchor_text: bool = False,
    ref_ids: Tensor | None = None,
) -> TorusGeometry:
    """`wrap` is (h, w). (False, True) is a cylinder, (False, False) is stock FLUX.2.
    The sequence is [txt, img] or [txt, img, ref]; only the img tokens live on the torus."""
    parts = (ctx_ids[:1], x_ids[:1]) if ref_ids is None else (ctx_ids[:1], x_ids[:1], ref_ids[:1])
    ids = torch.cat(parts, dim=1)  # [1, N, 4] of (t, h, w, l)
    num_txt = ctx_ids.shape[1]
    index = torch.arange(ids.shape[1], device=ids.device)
    is_txt = index < num_txt
    is_img = (index >= num_txt) & (index < num_txt + x_ids.shape[1])
    img_img = is_img[:, None] & is_img[None, :]

    ids_copy = ids.clone()
    use_copy = []
    for axis, n, do_wrap in ((1, grid[0], wrap[0]), (2, grid[1], wrap[1])):
        # The whole idea, on the n coordinates of one axis.
        r = torch.arange(n, device=ids.device)
        d = r[None, :] - r[:, None]  # [query, key]: key - query
        d_near = (d + n // 2) % n - n // 2 if do_wrap else d
        copy = torch.where(2 * r < n, r + n, r - n)  # the only other copy of a key ever wanted
        use = d_near != d
        assert torch.equal(torch.where(use, copy[None, :] - r[:, None], d), d_near)

        # From coordinates to tokens. Only img-img pairs wrap; text (at coordinate 0) and ref never do.
        pos = ids[0, :, axis]
        on_grid = pos.clamp(max=n - 1)  # a ref image may be larger than the grid; it is masked out anyway
        ids_copy[0, :, axis] = torch.where(is_img, copy[on_grid], pos)
        use_copy.append(use[on_grid][:, on_grid] & img_img)

    edges = [sum(model.pe_embedder.axes_dim[:i]) for i in range(5)]
    return TorusGeometry(
        pe=model.pe_embedder(ids),
        pe_copy=model.pe_embedder(ids_copy),
        use_copy_h=use_copy[0],
        use_copy_w=use_copy[1],
        txt_pair=is_txt[:, None] | is_txt[None, :],
        dims={name: slice(a, b) for name, a, b in zip("thwl", edges, edges[1:])},
        num_txt=num_txt,
        unanchor_text=unanchor_text,
    )


def torus_attention(q: Tensor, k: Tensor, v: Tensor, geo: TorusGeometry, q_chunk: int = 512) -> Tensor:
    """q, k, v: [B, heads, N, head_dim] *before* RoPE, sequence [txt, img(, ref)]. Returns [B, N, heads * head_dim].

    Peak memory is about eight float32 tensors of [B, heads, q_chunk, N]; lower q_chunk if it does not fit.
    """
    q = q * q.shape[-1] ** -0.5
    q_rot, k_rot = apply_rope(q, k, geo.pe)
    # h and w live in separate head dims, so one rotation holds both copies: read the h dims of
    # k_copy for the h copy, the w dims for the w copy. (The rotated q that comes with it is unused.)
    _, k_copy = apply_rope(q, k, geo.pe_copy)

    def dot(a: Tensor, b: Tensor, axis: str) -> Tensor:
        """The part of the logits that comes from one axis' block of the head dim."""
        return (a[..., geo.dims[axis]] @ b[..., geo.dims[axis]].transpose(-1, -2)).float()

    out = []
    for start in range(0, q.shape[2], q_chunk):
        rows = slice(start, start + q_chunk)
        qr = q_rot[:, :, rows]
        hw = torch.where(geo.use_copy_h[rows], dot(qr, k_copy, "h"), dot(qr, k_rot, "h"))
        hw = hw + torch.where(geo.use_copy_w[rows], dot(qr, k_copy, "w"), dot(qr, k_rot, "w"))
        if geo.unanchor_text:
            # Unrotated q . k is displacement zero. txt-txt pairs had that anyway (all text is at h=w=0).
            unrotated = dot(q[:, :, rows], k, "h") + dot(q[:, :, rows], k, "w")
            hw = torch.where(geo.txt_pair[rows], unrotated, hw)
        logits = dot(qr, k_rot, "t") + hw + dot(qr, k_rot, "l")
        out.append(torch.softmax(logits, dim=-1).to(v.dtype) @ v)
    return rearrange(torch.cat(out, dim=2), "b h n d -> b n (h d)")


def torus_forward(
    model: Flux2,
    x: Tensor,
    timesteps: Tensor,
    ctx: Tensor,
    guidance: Tensor | None,
    geo: TorusGeometry,
    q_chunk: int = 512,
) -> Tensor:
    """`Flux2.forward` (model.py:115) with the attention swapped. No weights change, none are added.
    `x` is the image tokens, followed by the reference tokens if the geometry was built with ref_ids."""
    num_txt = geo.num_txt

    vec = model.time_in(timestep_embedding(timesteps, 256))
    if model.use_guidance_embed:
        vec = vec + model.guidance_in(timestep_embedding(guidance, 256))

    double_block_mod_img = model.double_stream_modulation_img(vec)
    double_block_mod_txt = model.double_stream_modulation_txt(vec)
    single_block_mod, _ = model.single_stream_modulation(vec)

    img = model.img_in(x)
    txt = model.txt_in(ctx)

    pe_ctx, pe_x = geo.pe[:, :, :num_txt], geo.pe[:, :, num_txt:]
    for block in model.double_blocks:
        # _prepare_qkv returns q, k, v unrotated; the pe it concatenates is geo.pe again, unused here.
        q, k, v, _, _, mods = block._prepare_qkv(
            img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt
        )
        attn = torus_attention(q, k, v, geo, q_chunk)
        img, txt = block._apply_residuals(img, txt, attn[:, num_txt:], attn[:, :num_txt], mods)

    img = torch.cat((txt, img), dim=1)
    for block in model.single_blocks:
        q, k, v, mlp, mod_gate = block._qkv(img, single_block_mod)
        img = block._out(img, torus_attention(q, k, v, geo, q_chunk), mlp, mod_gate)

    img = img[:, num_txt:, ...]
    return model.final_layer(img, vec)


def denoise_torus(
    model: Flux2,
    img: Tensor,  # [P, N_img, C] noise, one latent per prompt
    txt: Tensor,  # [3P, N_txt, D]: P empty prompts, P prompts, P prompts + TORUS_PROMPT  (or [2P]: no third block)
    geo: TorusGeometry,
    timesteps: list[float],
    guidance: float,
    geo_guidance: float,
    q_chunk: int = 512,
    ref: Tensor | None = None,  # [1, N_ref, C] clean reference tokens, seen by every branch
    keep: Tensor | None = None,  # [1, N_img, 1] bool: tokens that must come out equal to `clean`
    clean: Tensor | None = None,  # [1, N_img, C] the encoded picture that `keep` refers to
    noise: Tensor | None = None,  # the pure noise `img` was built from; defaults to `img` itself
    on_step=None,  # called as on_step(steps_done, steps_total); the web page's progress bar
) -> Tensor:
    """Euler flow matching with the three-way guidance of reference_code/pipeline_flux2_erp.py:

        v = v_uncond + guidance * (v_cond - v_uncond) + geo_guidance * (v_geo - v_cond)

    The first difference is what the prompt adds over no prompt, the second what the torus sentence
    adds over the prompt alone. With geo_guidance == guidance the v_cond terms cancel and this is
    plain CFG on the long prompt, so the third branch only says something new away from that value.

    `keep`: the flow is x_t = (1 - t) x_0 + t noise, so where x_0 is known, x_t is known at every t.
    The noise used is the token's own starting noise, which keeps the kept region on one straight
    trajectory (at t=1 it is pure noise, at t=0 it is exactly `clean`).

    `noise` only has to be given when `img` is not pure noise: starting at timesteps[0] < 1 the
    caller hands in x_t = (1 - t) clean + t noise, and the kept region has to be re-noised with the
    same `noise` it started on, not with that mixture (see torus_generate.generate, `t_start`).
    """
    noise, num_img = img if noise is None else noise, img.shape[1]
    for step, (t_curr, t_prev) in enumerate(tqdm(list(zip(timesteps[:-1], timesteps[1:])), desc="denoise")):
        t_vec = torch.full((txt.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        branches = txt.shape[0] // img.shape[0]
        x = img.repeat(branches, 1, 1)
        if ref is not None:
            x = torch.cat((x, ref.expand(x.shape[0], -1, -1)), dim=1)
        pred = torus_forward(model, x, t_vec, txt, guidance=None, geo=geo, q_chunk=q_chunk)[:, :num_img]
        v_uncond, v_cond, *v_geo = pred.chunk(branches)
        v = v_uncond + guidance * (v_cond - v_uncond)
        if v_geo:
            v = v + geo_guidance * (v_geo[0] - v_cond)
        img = img + (t_prev - t_curr) * v
        if keep is not None:
            img = torch.where(keep, (1 - t_prev) * clean + t_prev * noise, img)
        if on_step is not None:
            on_step(step + 1, len(timesteps) - 1)
    return img


def decode_torus(
    ae, z: Tensor, wrap: tuple[bool, bool] = (True, True), pad: int = DECODE_PAD_TOKENS
) -> Tensor:
    """z: [B, 128, gh, gw] token grid. The decoder's convs zero-pad, which would put the seam back at
    the pixel level, so hand it the torus unrolled a little past each edge and crop what it returns."""
    ph, pw = pad * wrap[0], pad * wrap[1]
    z = F.pad(z, (pw, pw, ph, ph), mode="circular")
    x = ae.decode(z)
    return x[..., 16 * ph : x.shape[-2] - 16 * ph, 16 * pw : x.shape[-1] - 16 * pw]
