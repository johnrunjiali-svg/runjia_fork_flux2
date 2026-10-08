"""Four-way seamless images from FLUX.2, the old-fashioned way: copy the edge tokens around.

The image is gh x gw tokens. Before every attention, the keys and values of the image are continued
past each edge with a band of circular copies, `band` tokens wide: the key grid becomes
(gh + 2 band) x (gw + 2 band), and the copy at grid position (i, j) is the image token at
(i mod gh, j mod gw) with its RoPE taken at (i, j) -- a position outside the image, negative or past
the end. The queries stay the gh x gw image tokens at their own positions. So a query on the left
edge, at w = 0, finds the right-edge token w = gw - 1 twice: once where it is, at displacement
gw - 1, and once as the copy at j = -1, at displacement -1, i.e. as its left neighbour. That is
what a torus looks like from the left edge, and the model draws accordingly.

    2 x 2 image [[A, B], [C, D]], band 1: the keys are the 4 x 4 grid on positions {-1, 0, 1, 2}^2

        D C D C
        B A B A          4 queries, 16 keys; the same letter is the same value and the
        D C D C          same unrotated key, only the position, hence the rotation, differs
        B A B A

A band of `band` tokens is all that is needed for a seam: what makes the two sides of a cut
disagree is that neither saw the other's neighbourhood, and the band hands each side the other's
first `band` rows or columns. band = gh = gw is the full 3 x 3 board of copies.

How this differs from flux2/torus.py. There, every (query, key) pair is seen exactly once, at the
nearest periodic displacement, and the attention has to be written out by hand because which copy of
a key is used depends on the query. Here the copies are extra keys, present for every image query at
once, so stock attention does it: gather the keys and values by source index, rotate the keys with
the copies' positions, and call F.scaled_dot_product_attention. Two things come with that and are
not decisions, they are the method:

- A key and its copies are separate softmax entries. A query near the left edge sees the right-edge
  token both far away and next door; a query in the middle sees the copies at displacements of about
  gw/2 + band, which are not useful but take their share of the softmax mass.
- Displacements grow: a query at the far edge sees a copy at gw - 1 + band, which this image size
  never shows but a larger image does. RoPE has no table to run off, so nothing breaks.

Only image queries see the copies. Text and reference tokens attend to the stock [txt, img, ref]
keys, so the only thing that changes for them is what the image tokens become. Two SDPA calls do
that without a mask (one call with a boolean mask would do the same and give up the flash kernel).

band = 0 is stock attention, to the bit. The AE decoder is the one from torus.py, handed the latent
continued past its edges, so that its zero padding does not put the seam back; band 0 therefore
is stock attention + circular decode, one change at a time along a sweep over band.
"""

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from .model import Flux2, apply_rope, timestep_embedding


@dataclass
class CopyGeometry:
    """Which token every key is and where it sits, for the sequence [txt, img(, ref)]. Built once per image."""

    pe_q: Tensor  # [1, 1, N, 64, 2, 2] stock rotations of the N tokens at their own positions
    pe_k: Tensor  # [1, 1, N + E, 64, 2, 2] the same N, then the E copies rotated to the copies' positions
    key_src: Tensor  # [N + E] the token each key is: arange(N), then the image token each copy is of
    key_ids: Tensor  # [N + E, 4] (t, h, w, l) of every key: what pe_k was made from, kept to look at
    img: slice  # the image queries in the sequence: they see all N + E keys
    other: Tensor  # index of the text and reference queries: they see the first N
    num_txt: int


def build_copy_geometry(
    model: Flux2,
    x_ids: Tensor,
    ctx_ids: Tensor,
    grid: tuple[int, int],
    band: int,
    ref_ids: Tensor | None = None,
) -> CopyGeometry:
    """The sequence is [txt, img] or [txt, img, ref]; only image tokens are copied, `band` tokens
    past every edge of the (gh, gw) = `grid`."""
    parts = (ctx_ids[:1], x_ids[:1]) if ref_ids is None else (ctx_ids[:1], x_ids[:1], ref_ids[:1])
    ids = torch.cat(parts, dim=1)  # [1, N, 4] of (t, h, w, l)
    num_txt, num_img = ctx_ids.shape[1], x_ids.shape[1]
    gh, gw = grid
    index = torch.arange(ids.shape[1], device=ids.device)
    img = slice(num_txt, num_txt + num_img)

    # The padded grid [-band, gh + band) x [-band, gw + band), minus the image itself.
    i, j = torch.meshgrid(torch.arange(-band, gh + band), torch.arange(-band, gw + band), indexing="ij")
    i, j = i.flatten().to(ids.device), j.flatten().to(ids.device)
    outside = (i < 0) | (i >= gh) | (j < 0) | (j >= gw)
    i, j = i[outside], j[outside]

    # The image token at (h, w), whatever order x_ids lists the grid in.
    token_at = torch.empty((gh, gw), dtype=torch.long, device=ids.device)
    token_at[ids[0, img, 1], ids[0, img, 2]] = index[img]
    src = token_at[i % gh, j % gw]
    copy_ids = ids[0, src].clone()
    copy_ids[:, 1], copy_ids[:, 2] = i, j  # the same token, at the position of its copy
    key_ids = torch.cat((ids[0], copy_ids))

    return CopyGeometry(
        pe_q=model.pe_embedder(ids),
        pe_k=model.pe_embedder(key_ids[None]),
        key_src=torch.cat((index, src)),
        key_ids=key_ids,
        img=img,
        other=torch.cat((index[:num_txt], index[num_txt + num_img :])),
        num_txt=num_txt,
    )


def rotate(x: Tensor, pe: Tensor) -> Tensor:
    """`apply_rope` for one tensor, so that queries and keys can be rotated with different positions."""
    return apply_rope(x, x, pe)[0]


def copy_attention(q: Tensor, k: Tensor, v: Tensor, geo: CopyGeometry) -> Tensor:
    """q, k, v: [B, heads, N, head_dim] *before* RoPE, sequence [txt, img(, ref)]. Returns [B, N, heads * head_dim]."""
    n = q.shape[2]
    q = rotate(q, geo.pe_q)
    k = rotate(k[:, :, geo.key_src], geo.pe_k)  # a copy is its source's key, rotated to where the copy sits
    v = v[:, :, geo.key_src]
    out = torch.empty_like(q)
    out[:, :, geo.img] = F.scaled_dot_product_attention(q[:, :, geo.img], k, v)
    out[:, :, geo.other] = F.scaled_dot_product_attention(q[:, :, geo.other], k[:, :, :n], v[:, :, :n])
    return rearrange(out, "b h n d -> b n (h d)")


def copy_forward(
    model: Flux2, x: Tensor, timesteps: Tensor, ctx: Tensor, guidance: Tensor | None, geo: CopyGeometry
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

    pe_ctx, pe_x = geo.pe_q[:, :, :num_txt], geo.pe_q[:, :, num_txt:]
    for block in model.double_blocks:
        # _prepare_qkv returns q, k, v unrotated; the pe it concatenates is pe_q again, unused here.
        q, k, v, _, _, mods = block._prepare_qkv(
            img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt
        )
        attn = copy_attention(q, k, v, geo)
        img, txt = block._apply_residuals(img, txt, attn[:, num_txt:], attn[:, :num_txt], mods)

    img = torch.cat((txt, img), dim=1)
    for block in model.single_blocks:
        q, k, v, mlp, mod_gate = block._qkv(img, single_block_mod)
        img = block._out(img, copy_attention(q, k, v, geo), mlp, mod_gate)

    img = img[:, num_txt:, ...]
    return model.final_layer(img, vec)


def denoise_copy(
    model: Flux2,
    img: Tensor,  # [P, N_img, C] noise, one latent per prompt
    txt: Tensor,  # [P, N_txt, D] the prompts; at guidance != 1, [2P, N_txt, D]: P empty prompts, then the prompts
    geo: CopyGeometry,
    timesteps: list[float],
    guidance: float = 1.0,
    ref: Tensor | None = None,  # [1, N_ref, C] clean reference tokens, seen by every branch
) -> Tensor:
    """Euler flow matching, `sampling.denoise` with the attention swapped (and `torus.denoise_torus`
    with the other attention). guidance = 1 is the distilled klein recipe, one pass per step; any
    other value is classifier-free guidance against the empty prompt, two passes."""
    num_img, branches = img.shape[1], txt.shape[0] // img.shape[0]
    assert branches == (1 if guidance == 1 else 2), f"{branches} text blocks for guidance {guidance}"
    for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:])), desc="denoise"):
        t_vec = torch.full((txt.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        x = img.repeat(branches, 1, 1)
        if ref is not None:
            x = torch.cat((x, ref.expand(x.shape[0], -1, -1)), dim=1)
        v = copy_forward(model, x, t_vec, txt, guidance=None, geo=geo)[:, :num_img]
        if branches == 2:
            v_uncond, v_cond = v.chunk(2)
            v = v_uncond + guidance * (v_cond - v_uncond)
        img = img + (t_prev - t_curr) * v
    return img
