"""Attention for Qwen-Image 2.1 with the target image on a torus. No weights change, none are added.

Qwen-Image 2.1's attention is block-causal over one joint sequence [text and condition images ...,
target image]: text is causal, every image block is bidirectional inside itself, and a token sees
everything before it. Two consequences shape this file.

1. Prefix tokens (text, condition images) never attend to the target image, so their rows are
   untouched by anything done to the target's positions: `prefix_attention` is the stock prefill,
   segment by segment, and the model's KV cache of those rows (keys stored *after* RoPE) stays valid.
2. Only the target image's rows change, and only in how their logits are formed. Against prefix keys
   the displacement is the stock one (or, with `unanchor_text`, the one the centre token has, for
   text keys only); against the target's own keys it is the torus rule of rope.py.

`torus_attention` writes the target rows' logits out by hand in query chunks, like
src/flux2/torus.py: the key copy a pair wants depends on the query, so no single rotated key tensor
serves every query and the fused kernel cannot be used. The frame axis never wraps and the h and w
axes live in separate head dims, so the logit is a sum of three per-axis partial products and
wrapping an axis only touches its own 56 dims.

In `mode="periodic"` without unanchoring nothing here is needed: `TorusRope` hands the transformer
a snapped rotary table and the stock processor runs, fused kernel and all. `install` makes that
choice from the config.
"""

import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
from torch import Tensor
from torch.nn import functional as F

from .rope import PeConfig, TorusGeometry, TorusRope


class TorusContext:
    """Shared by every block's processor and set by the sampler before each forward: the geometry of
    the target image (one per generation) and which prefix rows are text (one per guidance branch,
    since every branch has its own prompt and so its own prefix)."""

    def __init__(self):
        self.geo: TorusGeometry | None = None
        self.prefix_is_text: Tensor | None = None


def rotate(x: Tensor, pe: Tensor) -> Tensor:
    """x: [B, S, heads, D] real, pe: [S, D/2] complex. apply_rotary_emb_qwen(use_real=False), with the
    table given per call: adjacent dims (2i, 2i+1) are plane i, rotated in float32."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * pe[None, :, None, :]).flatten(3).to(x.dtype)


def prefix_attention(q: Tensor, k: Tensor, v: Tensor, segments, key_valid: Tensor | None) -> Tensor:
    """The stock block-causal prefill over the prefix rows (QwenImage21AttnProcessor, prefill branch):
    every segment attends to the keys before it plus its own block, text segments causally within
    their own keys. q, k, v: [B, P, heads, D] after RoPE. Returns [B, P, heads, D]."""
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    outs = []
    for start, end, is_text in segments:
        mask = None
        if is_text:
            n = end - start
            ones = torch.ones(n, start, dtype=torch.bool, device=q.device)
            mask = torch.cat([ones, torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))], 1)[
                None, None
            ]
        if key_valid is not None:
            valid = key_valid[:, None, None, :end]
            mask = valid if mask is None else mask & valid
        outs.append(
            F.scaled_dot_product_attention(
                qh[:, :, start:end], kh[:, :, :end], vh[:, :, :end], attn_mask=mask
            )
        )
    return torch.cat(outs, dim=2).transpose(1, 2)


def torus_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_prefix: Tensor,
    v_prefix: Tensor,
    pe: Tensor,
    geo: TorusGeometry,
    prefix_is_text: Tensor | None,
    key_valid: Tensor | None = None,
    q_chunk: int = 512,
) -> Tensor:
    """The target image's rows. q, k, v: [B, N_t, heads, D] *before* RoPE; k_prefix, v_prefix:
    [B, P, heads, D], keys already rotated by the stock table (what the KV cache holds); pe: [N_t, D/2]
    complex, the rotary table of the target rows for this branch (frame planes included). Returns
    [B, N_t, heads, D]. Peak memory is a handful of float32 [B, heads, q_chunk, P + N_t] tensors."""
    B, N, H, D = q.shape
    q = q * D**-0.5
    f = geo.planes["f"]
    pe_copy = torch.cat([pe[:, f], geo.pe_copy_hw], dim=-1)
    q_rot, k_rot = rotate(q, pe).transpose(1, 2), rotate(k, pe).transpose(1, 2)  # [B, H, N, D]
    k_copy = rotate(k, pe_copy).transpose(1, 2) if geo.any_copy else None
    q_un = None
    if geo.cfg.unanchor_text:
        # The centre token's rotation: frame as everyone, no turn on h and w. Against a text key at p
        # that is the displacement (p, p) whatever (h, w) the query sits at.
        pe_un = torch.cat([pe[:, f], torch.ones_like(pe[:, f.stop :])], dim=-1)
        q_un = rotate(q, pe_un).transpose(1, 2)
        assert prefix_is_text is not None, "unanchor_text needs to know which prefix rows are text"
    kp = k_prefix.transpose(1, 2)
    v_all = torch.cat([v_prefix, v], dim=1).transpose(1, 2)
    invalid = None if key_valid is None else ~key_valid[:, None, None, :]

    def dot(a: Tensor, b: Tensor, axis: str) -> Tensor:
        s = geo.dims[axis]
        return (a[..., s] @ b[..., s].transpose(-1, -2)).float()

    out = []
    for start in range(0, N, q_chunk):
        rows = slice(start, start + q_chunk)
        qr = q_rot[:, :, rows]
        lp = (qr @ kp.transpose(-1, -2)).float()
        if q_un is not None:
            lp = torch.where(prefix_is_text, (q_un[:, :, rows] @ kp.transpose(-1, -2)).float(), lp)
        lt = dot(qr, k_rot, "f")
        for axis, use in (("h", geo.use_copy_h), ("w", geo.use_copy_w)):
            if use is None:
                lt = lt + dot(qr, k_rot, axis)
            else:
                lt = lt + torch.where(use[rows], dot(qr, k_copy, axis), dot(qr, k_rot, axis))
        logits = torch.cat([lp, lt], dim=-1)
        if invalid is not None:
            logits = logits.masked_fill(invalid, float("-inf"))
        out.append(torch.softmax(logits, dim=-1).to(v_all.dtype) @ v_all)
    return torch.cat(out, dim=2).transpose(1, 2)


class TorusAttnProcessor:
    """QwenImage21AttnProcessor with the target rows replaced by `torus_attention`. Same call signature,
    same KV-cache protocol (prefix keys stored after RoPE on `extract`, read on `cached`)."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, ctx: TorusContext):
        self.ctx = ctx

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        attention_mask=None,
        rotary_emb: Tensor | None = None,
        layer_cache=None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments=None,
        key_valid: Tensor | None = None,
    ) -> Tensor:
        geo = self.ctx.geo
        assert geo is not None, "set TorusContext.geo before running the transformer"
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query).to(value.dtype)
        key = attn.norm_k(key).to(value.dtype)
        n_t = geo.num_target
        # The padding mask reaches the processor as `key_valid` [B, S] in prefill and as a broadcastable
        # `attention_mask` in decode; either way it is one bool per key.
        valid = key_valid
        if valid is None and torch.is_tensor(attention_mask):
            valid = attention_mask.reshape(hidden_states.shape[0], -1)

        if kv_cache_mode == "cached":
            k_prefix, v_prefix = layer_cache.get()
            out_prefix = None
            q_t, k_t, v_t = query, key, value
        else:
            prefix = query.shape[1] - n_t
            pe_prefix = rotary_emb[:prefix]
            q_prefix, k_prefix = rotate(query[:, :prefix], pe_prefix), rotate(key[:, :prefix], pe_prefix)
            v_prefix = value[:, :prefix]
            if kv_cache_mode == "extract":
                layer_cache.store(k_prefix.clone(), v_prefix.clone())
            out_prefix = prefix_attention(q_prefix, k_prefix, v_prefix, segments, valid)
            q_t, k_t, v_t = query[:, prefix:], key[:, prefix:], value[:, prefix:]

        out_t = torus_attention(
            q_t,
            k_t,
            v_t,
            k_prefix,
            v_prefix,
            rotary_emb[-n_t:],
            geo,
            self.ctx.prefix_is_text,
            valid,
            geo.cfg.q_chunk,
        )
        out = out_t if out_prefix is None else torch.cat([out_prefix, out_t], dim=1)
        out = out.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](out))


def install(transformer, ctx: TorusContext, cfg: PeConfig) -> TorusRope:
    """Put the rope wrapper in (once) and pick the processor the config needs. Returns the wrapper."""
    if not isinstance(transformer.pos_embed, TorusRope):
        transformer.pos_embed = TorusRope(transformer.pos_embed)
    transformer.pos_embed.cfg = cfg
    processor = TorusAttnProcessor(ctx) if cfg.needs_manual_attention else QwenImage21AttnProcessor()
    transformer.set_attn_processor(processor)
    return transformer.pos_embed
