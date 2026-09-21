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

Text-to-image only: reference-image tokens are not handled.
"""

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from .model import Flux2, apply_rope, timestep_embedding

# Appended to the prompt for the third guidance branch, the analogue of ERP_PROMPT in
# reference_code/erp_utils.py. Untuned.
TORUS_PROMPT = (
    "Seamless tileable image, a perfectly periodic pattern that repeats in all four directions, "
    "the left edge continues into the right edge and the top edge continues into the bottom edge, "
    "uniform composition with no center and no horizon, no border, no frame, no vignette."
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
    img_img: Tensor  # [N, N] bool: both tokens are image tokens
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
) -> TorusGeometry:
    """`wrap` is (h, w). (False, True) is a cylinder, (False, False) is stock FLUX.2."""
    ids = torch.cat((ctx_ids[:1], x_ids[:1]), dim=1)  # [1, N, 4] of (t, h, w, l)
    num_txt = ctx_ids.shape[1]
    is_img = torch.arange(ids.shape[1], device=ids.device) >= num_txt
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

        # From coordinates to tokens. Text sits at coordinate 0 and never wraps.
        pos = ids[0, :, axis]
        ids_copy[0, :, axis] = torch.where(is_img, copy[pos], pos)
        use_copy.append(use[pos][:, pos] & img_img)

    edges = [sum(model.pe_embedder.axes_dim[:i]) for i in range(5)]
    return TorusGeometry(
        pe=model.pe_embedder(ids),
        pe_copy=model.pe_embedder(ids_copy),
        use_copy_h=use_copy[0],
        use_copy_w=use_copy[1],
        img_img=img_img,
        dims={name: slice(a, b) for name, a, b in zip("thwl", edges, edges[1:])},
        num_txt=num_txt,
        unanchor_text=unanchor_text,
    )


def torus_attention(q: Tensor, k: Tensor, v: Tensor, geo: TorusGeometry, q_chunk: int = 512) -> Tensor:
    """q, k, v: [B, heads, N, head_dim] *before* RoPE, sequence [txt, img]. Returns [B, N, heads * head_dim].

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
            hw = torch.where(geo.img_img[rows], hw, unrotated)
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
    """`Flux2.forward` (model.py:115) with the attention swapped. No weights change, none are added."""
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
    img: Tensor,  # [1, N_img, C]
    txt: Tensor,  # [3, N_txt, D]: empty prompt, prompt, prompt + TORUS_PROMPT  (or the first two only)
    geo: TorusGeometry,
    timesteps: list[float],
    guidance: float,
    geo_guidance: float,
    q_chunk: int = 512,
) -> Tensor:
    """Euler flow matching with the three-way guidance of reference_code/pipeline_flux2_erp.py:

        v = v_uncond + guidance * (v_cond - v_uncond) + geo_guidance * (v_geo - v_cond)

    The first difference is what the prompt adds over no prompt, the second what the torus sentence
    adds over the prompt alone. With geo_guidance == guidance the v_cond terms cancel and this is
    plain CFG on the long prompt, so the third branch only says something new away from that value.
    """
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((txt.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = torus_forward(
            model, img.expand(txt.shape[0], -1, -1), t_vec, txt, guidance=None, geo=geo, q_chunk=q_chunk
        )
        v_uncond, v_cond = pred[0:1], pred[1:2]
        v = v_uncond + guidance * (v_cond - v_uncond)
        if txt.shape[0] == 3:
            v = v + geo_guidance * (pred[2:3] - v_cond)
        img = img + (t_prev - t_curr) * v
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
