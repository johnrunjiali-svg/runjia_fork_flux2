# Qwen-Image 2.1: how it works, and how we put it on a torus

Written from the code that runs it (`diffusers` at commit `80c7ed2`, files
`pipelines/qwenimage21/pipeline_qwenimage21.py`, `models/transformers/transformer_qwenimage21.py`,
`models/autoencoders/autoencoder_kl_qwenimage21.py`) and the checkpoint's configs on the Hub
(`Qwen/Qwen-Image-2.1`). Numbers are for that checkpoint. The second half is what `src/qwen_torus`
changes and why.

## 1. What is in the checkpoint

| component | class | size | role |
|---|---|---|---|
| `text_encoder` | `Qwen3VLForConditionalGeneration` | 36 text layers, hidden 4096 (~8B); 27-layer ViT | reads the prompt *and* the condition images; its last hidden state is the conditioning |
| `processor` | `Qwen3VLProcessor` | | chat template, tokenizer, image patching for the VLM |
| `transformer` | `QwenImage21Transformer2DModel` | 32 single-stream blocks, 32 heads × 128, hidden 4096 (~7B) | the denoiser |
| `vae` | `AutoencoderKLQwenImage21` | 64 latent channels, 16× spatial, RGBA in and out | pixels ↔ latents |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | exponential shift, terminal 0.02 | the noise schedule |

In bf16 that is roughly 16 GB + 14 GB + 1 GB of weights. On a 48 GB card the three do not fit next to
the activations of a 2048² image, hence `enable_model_cpu_offload()`: each model is moved to the GPU
when it is called and back when the next one is, so at most one is resident. `QwenTorusPipe` does
this by default.

## 2. Data flow, first principles

### 2.1 Text (and condition images) → conditioning

The prompt is not passed through `apply_chat_template`; it is dropped into a raw template string the
checkpoint was trained with:

```
<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n
<|im_start|>user\n{prompt}<|im_end|>\n
<|im_start|>assistant\n
```

With condition images, each one is spliced into the user turn as
`<image1><|vision_start|><|image_pad|><|vision_end|>` before the prompt text. The processor expands
each `<|image_pad|>` into one placeholder token per **vision slot**: a Qwen3-VL slot is a 2×2 merge of
16-px patches, i.e. **32×32 pixels**, so an image of `H×W` pixels yields `(H/32)(W/32)` slots. The
VLM runs (vision tower → merger → 36 decoder layers) and the output is the **last decoder layer
before the final RMSNorm** — the pipeline hooks the norm out, because `transformers ≥ 5` would
otherwise return the normalised state. The leading system-turn tokens are dropped (`_drop_idx`), so
the conditioning is `[L, 4096]` for the user turn onwards, plus a `[L]` bool `image_pad_mask`
marking the slot positions. An RGBA condition image is flattened over white for the VLM only.

An empty prompt is encoded as `" "` (Qwen has no BOS token). The pipeline uses true CFG with a negative
prompt when `true_cfg_scale > 1`; the model card samples **without guidance**.

### 2.2 Pixels → latents

The VAE is a Wan-style causal 3D autoencoder specialised to one frame: 2D convolutions with zero
padding, five stages with channel multipliers `[1, 2, 4, 8, 8]` (base 96 encoder / 144 decoder),
four 2× resamplings → **16× spatial**, latent **64 channels**. It takes and returns **4 channels
(RGBA)**: the model can draw transparent images, which is why `pipe(...)` returned an RGBA PIL image.
Latents are normalised per channel, `(z − mean) / std`, with 64 constants from the config. Both
encoder and decoder have a mid block with two residual blocks and **one global self-attention over
the latent grid** (as the FLUX VAE does).

A 1024² picture is `64×64` latent pixels. The transformer is **unpatched** (`patch_size = 1`): one
token per latent pixel, i.e. **one token per 16×16 pixels**, 4096 tokens at 1024², 16 384 at 2048².
Packing is a plain flatten, `[B, 64, h, w] → [B, h·w, 64]`.

### 2.3 The joint sequence

The transformer sees **one** sequence. It is built from the VLM sequence by expanding every vision
slot four-fold (a 32×32 slot is 2×2 latent tokens) and overwriting those positions with the VAE
latents of the corresponding condition image; the target image's `N_t` tokens are appended at the
end (the pipeline appends `N_t/4` slots to `image_pad_mask`). So for text-to-image the sequence is
`[text (L), target (N_t)]` and for editing `[text, cond-image tokens, text, ..., target]`, in the
order the template put them. Note the VLM's own image embeddings at the slot positions are
**discarded** — the DiT sees VAE latents there; what the VLM understood of the picture reaches the
DiT only through the text tokens that attended to it.

Inputs: `img_in: Linear(64 → 4096)` on the latents, `txt_in: RMSNorm → Linear(4096 → 4096) → GELU →
Linear` on the embeddings (`QwenImage21TextProjection`).

### 2.4 Position: three-axis complex RoPE (`QwenImage21Rope`)

Each 128-dim head is 64 complex planes in three groups, `axes_dims_rope = (16, 56, 56)` → **(8 | 28
| 28) planes for (frame | height | width)**, plane `m` of an axis turning by `θ^{-m/(d/2)} · p` for
coordinate `p`, `θ = 10000`, `d = 16` or `56`. Adjacent real dims `(2i, 2i+1)` are plane `i`;
rotation is a complex multiply in fp32 (`apply_rotary_emb_qwen(use_real=False)`). The table is
precomputed for positions `−1024 … 8191`.

Coordinates are assigned by walking the joint sequence with one running `position`:

- a **text token** gets `position` on all three axes, then `position += 1`;
- an **image block** of `h×w` tokens gets `frame = position` for every token, and `height ∈ [−(h −
  h//2), h//2)`, `width ∈ [−(w − w//2), w//2)` on a grid **centred on zero**; then `position += max(h, w)`.

So the target image's tokens all share one frame coordinate `f` (= text length + earlier images'
extents) and differ only in `(h, w)`; a text token at `p` is at `(p, p, p)`; a condition image's
tokens are at `(f', h, w)` with the same centred grid, so token `(h, w)` of a reference is at zero
spatial displacement from token `(h, w)` of the output — the geometry editing relies on.

Because RoPE is relative, the logit between tokens `p` and `q` is `Σ_axes x_pᵀ R_axis(q − p) x_q`:
only the per-axis displacement matters, and the three axes are block-diagonal.

### 2.5 Attention: block-causal, single stream

Every block is `x += tanh(gate₁) · Attn(LN(x)(1 + scale₁)); x += tanh(gate₂) · SwiGLU(LN(x)(1 +
scale₂))` with q/k RMSNorm per head. There is no separate text stream: one QKV projection over the
joint sequence. The mask is

```
allowed(q, k) = (q ≥ k) or same_image_block(q, k)
```

i.e. **the sequence is causal, except that every image block is bidirectional inside itself**. Text
attends only to what precedes it; a condition image attends to the text before it and to itself; the
target image — last — attends to everything. Two consequences:

- Prefix rows (text + condition images) **never see the target**, so their activations do not
  depend on the noisy latent.
- With `causal_condition = True`, prefix tokens are modulated from **t = 0** (an extra row of the
  shared modulation; `_select_modulation_rows`) rather than from the sampled timestep. Together these
  make the prefix activations identical at every step, so the pipeline **prefills a KV cache** of the
  prefix (keys stored *after* RoPE) on step 0 (`kv_cache_mode="extract"`) and on later steps
  (`"cached"`) runs only the target rows against `[cached prefix K/V, target K/V]`.

Two processors implement the mask: `QwenImage21FlexAttnProcessor` (one `flex_attention` call with a
`BlockMask`; needs `torch.compile`, or it materialises dense fp32 scores) and the default
`QwenImage21AttnProcessor`, which decomposes the prefill into one SDPA call per prefix segment plus one
for the target. Padded prompt positions are masked out as keys.

Modulation is **shared**: one `SiLU → Linear(4096 → 4·4096)` on the timestep embedding, and every
block slices the same `[scale₁, gate₁, scale₂, gate₂]`. Output: `AdaLN (scale only) → Linear(4096 →
64)`; the pipeline keeps the last `N_t` rows.

### 2.6 Sampling

Rectified flow: `x_t = (1 − t) x₀ + t ε`, the network predicts the velocity, Euler steps
`x ← x + (t_{i+1} − t_i) v`. The schedule for `n` steps is `σ = linspace(1, 1/n, n)`, bent by the
resolution-dependent exponential shift `s(u) = e^μ / (e^μ + 1/u − 1)` with `μ` linear in the target
token count (`0.5` at 256 tokens → `0.9` at 8192), then **stretched so the last σ is 0.02**
(`shift_terminal`), then `0` appended. The timestep handed to the network is `σ` itself (the
pipeline passes `t/1000`). Default 40 steps; true CFG optional.

### 2.7 Latents → pixels

Unpack to `[1, 64, 1, h, w]`, undo the normalisation, decode (the decoder is clamped to `[−1, 1]`),
take frame 0 → `[1, 4, 16h, 16w]` RGBA.

## 3. Differences from FLUX.2 that matter for us

| | FLUX.2 (klein) | Qwen-Image 2.1 |
|---|---|---|
| streams | double-stream blocks then single-stream | single stream throughout |
| attention mask | full | block-causal; prefix never sees the image |
| text position | all text at `(0, 0)` on h/w | text at `(p, p, p)`, advancing |
| image grid origin | `0 … n−1` | centred: `−n/2 … n/2−1` |
| RoPE axes | `(t, h, w, l)` = 32 dims each, real 2×2 blocks | `(f, h, w)` = (16, 56, 56) dims, complex planes |
| θ | 2000 | 10000 |
| token | 2×2 latent pixels (patched) | 1 latent pixel; sides must be multiples of 32 |
| guidance | distilled (klein) or embedded | true CFG, 3 full evaluations for 3 branches |
| references | appended `[txt, img, ref]`, told apart by the `t` axis | interleaved into the prompt, told apart by the frame axis |
| VAE | RGB, 16 latent channels ×2×2 patch | RGBA, 64 channels, 16× |
| KV cache | our own kv-cache code | built in |

## 4. Putting the target image on a torus (`src/qwen_torus`)

### 4.1 What changes and what does not

Only the **target rows' logits** change. By (2.5) the prefix rows cannot see the target, so they are
computed by the stock rule and their KV cache stays valid. Within the target rows, pairs with
**prefix keys** keep the stock displacement (or, with `unanchor_text`, the one the centre token has,
for text keys only — references are left alone so an edit still lines up with its reference), and
pairs with **target keys** use the torus rule. Weights are untouched.

### 4.2 Mode `nearest`: the displacement wraps (`torus.py`, moved to a centred grid)

On a circle of `n` tokens the displacement from `p` to `q` is its nearest periodic copy,
`d_near = ((q − p + n/2) mod n) − n/2`. Every query sees itself in the middle of the image, and every
rotation used is one the model met in training. `d_near ∈ {d − n, d, d + n}`, so `R(d_near) = R(p)ᵀ
R(q + s·n)`: queries are rotated as always and keys come in two copies, at `q` and at `q ± n` (lower
half of the axis: `+n`; upper half: `−n`). Which copy a pair uses depends on the query, so the logits
are written out by hand in query chunks (`attention.torus_attention`): per axis, `where(use_copy,
q·k_copy, q·k)` over that axis's 56 dims, plus the frame axis's 16. About 1.9× the QK work of stock
attention, and the fp32 logits of a `[32 heads, q_chunk, N]` chunk in memory — fine at 1024²
(4096 tokens), slow at 2048² (16 384 tokens; see 4.5).

The centred grid changes nothing in the rule (only `q − p` enters); `build_geometry` shifts
coordinates to `0 … n−1` for the half-plane test and back.

### 4.3 Mode `periodic`: the frequencies wrap

A plane with frequency `ω` is periodic on `n` tokens iff `ω n = 2πk` for an integer `k`: the plane
makes a whole number of turns across the image. `k_m = ω_m n / 2π` is the plane's **number of
cycles**; for `θ = 10000`, `d = 56`: plane `m` has `ω_m = 10000^{−m/28}`, so on a 64-token side
`k = 10.2, 7.3, 5.3, 3.8, 2.7, 2.0, 1.4, 1.0, 0.73, …` — **8 planes make at least one turn**, 20 do
not; on 128 tokens, 10 do. `PeConfig` decides:

- `min_cycles` (default 1): planes with `k ≥ min_cycles` are **snapped** to `round(k)` turns
  (`rounding = nearest | floor | ceil`), `ω' = 2π round(k) / n`. This is "where high meets low": lower
  it below 1 to stretch nearly-periodic slow planes to one turn, raise it to touch fewer planes.
- `max_rel_error` (off by default): additionally require `|k − round k| / k ≤ max_rel_error`, i.e.
  snap only planes already close to periodic — the criterion of `reference_code/erp_utils.py`.
- `low_freq`: what happens to the planes below the threshold. `keep` leaves them as trained (the
  default: they turn less than once, so the seam discontinuity in their phase is `2π − ωn`);
  `fundamental` stretches them to exactly one turn; `zero` drops them (constant).
- `scope`: `image` snaps only the target image's rows (prefix tokens keep the stock table; a
  target↔text pair then mixes two tables, `ω p − ω' h`, which is what the ERP reference does for
  FLUX); `all` applies the snapped table to every token.

The rotation stays per token, so `TorusRope` just hands the transformer a different table and the
**stock fused processor runs** — no extra cost. Only `unanchor_text` forces the hand-written path.

`mode="both"` runs the nearest-copy rule on top of the snapped table (the low planes get the
displacement rule, the snapped planes are periodic already).

### 4.4 Text anchoring

An image token at `(h, w)` sees a text token at `p` through displacement `(p − f, p − h, p − w)`: it
knows where it is. `unanchor_text=True` rotates the query by `(f, 0, 0)` against text keys, so every
image token sees the text as the centre token does, and the network is **exactly equivariant to
cyclic shifts** of the latent in `nearest` mode and in `periodic` mode when every plane is periodic
(`scripts/qwen_selftest.py` checks both; with `low_freq="keep"` it is not, as expected). Condition
images are never unanchored.

### 4.5 Sampling, guidance, decoding

`generate` runs the stock schedule (reproduced in `schedule`, equal to the scheduler to 1e-5, and
invertible so `t_start < 1` starts exactly there), Euler in fp32, and three branches

```
v = v_uncond + guidance (v_cond − v_uncond) + geo_guidance (v_geo − v_cond)
  = (1 − guidance) v_uncond + (guidance − geo_guidance) v_cond + geo_guidance v_geo
```

each a **separate forward pass at batch 1 with its own KV cache** (prompts differ in length). A branch
with weight 0 is not encoded and not run: `guidance = 1` (the model card's setting) drops the
unconditional one, `geo_guidance = 0` the torus sentence. `keep` / `t_start` work as in `torus.py`.

Decoding pads the latent **circularly by 18 tokens** (the decoder's local receptive field: `conv_in`,
the mid block's 4 convs and the first up block's 6 at latent resolution ≈ 11, then ≈ 3 + 1.5 + 0.75
+ 0.4 from the later stages, + the resample convs — `measure_decoder_receptive_field` measures it on
the real VAE) and crops. The mid-block attention is global, so this is an approximation, as it was
for FLUX. The RGBA output is composited over white.

Cost at 1024² (4096 tokens): the hand-written attention holds `32 × 512 × (4096 + L)` fp32 logits
per chunk, a few hundred MB; with cpu offload the transformer is moved in once per request. At 2048²
(16 384 tokens) the same tensors are 16× larger and the fp32 traffic dominates: use `periodic` mode
(fused) there, or lower `q_chunk`, and `--vae_tiling` for the decoder. A `flex_attention` version of
the nearest-copy rule (four rotated key copies, a block mask selecting one per pair) would bring it
back to a fused kernel; not done.

## 5. Running it

```
PYTHONPATH=src uv run python scripts/qwen_selftest.py              # CPU, seconds, toy weights
PYTHONPATH=src uv run python scripts/qwen_web.py                   # the page, port 7862 (ssh -L 7862:localhost:7862)
PYTHONPATH=src uv run python scripts/qwen_web.py --fake            # the page on toy weights, no GPU
PYTHONPATH=src uv run python scripts/qwen_cli.py --prompt "..." --pe '{"mode":"periodic","min_cycles":0.5}'
PYTHONPATH=src uv run python scripts/qwen_cli.py --measure_vae     # the decoder's receptive field, once
```

`PeConfig` fields (`src/qwen_torus/rope.py`): `mode`, `wrap_h`, `wrap_w`, `unanchor_text`,
`min_cycles`, `max_rel_error`, `rounding`, `low_freq`, `scope`, `q_chunk`. `describe_tables(rope,
cfg, grid)` prints what a config does to every plane before spending GPU time.

**Seamless off is stock.** With both wraps off (`STOCK`, what the page sends when the seamless box is
unticked) `cfg.nearest` and `cfg.periodic` are both false: the rope wrapper returns the stock table
and `install` puts the stock fused processor back, whatever `mode` says. Only `unanchor_text` still
forces the hand-written path. `generate` restores that state when it returns (or fails), so
`pipe.pipe(...)` — the untouched `QwenImage21Pipeline` — is always available for an A/B. What
`generate` with `STOCK` and `geo_guidance=0` still does differently from `pipe.pipe(...)`: it keeps
the latent in fp32 between Euler steps (the stock pipeline rounds to bf16 every step), resizes the
first condition image to the output size instead of to a 1024²-area box, and composites the RGBA
result over white. Same seed gives the same initial noise.

## 6. Managing experiments

The unit of an experiment is a **position-encoding config**, and it is deliberately separate from
the sampling settings (prompt, seed, steps, guidance, seamless strength, size), which are the same
across a comparison.

- **`configs/pe/*.json`** are the named variants: one file per idea, a `note` saying what it tests,
  only the fields that differ from the defaults. Add a file to add a variant; nothing else changes.
  The ones there now: `stock`, `nearest`, `nearest_unanchored`, `periodic_keep`,
  `periodic_keep_half`, `periodic_fundamental`, `periodic_zero`, `periodic_strict`,
  `periodic_all_tokens`, `both`.
- **A config can be given four ways**, all through `PeConfig.load`: a file path (`--pe
  configs/pe/periodic_keep.json`), a json string (`--pe '{"mode":"periodic","min_cycles":0.5}'`), a
  dict / `PeConfig(...)` in Python, or the preset buttons in the web page, which fill the fields and
  can then be edited by hand.
- **Sweeps**: `scripts/qwen_cli.py --sweep configs/pe --prompts prompts/x.txt --seed 0` runs every
  prompt under every preset with the same seed, named `<slug>_<seed>_<preset>.png`, so the pictures
  of one prompt sit next to each other in the folder and differ only in the config.
- **Records**: every CLI image has a `.json` next to it, and every web run has `run.json` in its
  folder, with the full resolved config (`pe`) plus every sampling setting, the reference, the mask
  and the outlines. A run is reproducible from that folder alone.
- **Before spending GPU time** on a `periodic` variant, `describe_tables` shows which planes it
  changes and to what, per axis, for the grid you are about to use.

What to vary first, in order of how much it changes: `mode` (nearest vs periodic), `unanchor_text`,
then inside periodic `min_cycles` / `low_freq` (which planes are made periodic and how), then
`scope`; `rounding` and `max_rel_error` are refinements. `q_chunk` only trades memory for speed.
