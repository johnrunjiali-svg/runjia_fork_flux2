# FLUX.2 Codebase — Reading & Development Notes

Working notes for the `runjia_fork_flux2` study effort.
Goal: full line-by-line understanding of this codebase, with emphasis on **positional
encoding**, as reference material for my own project.

Repo state at session 1: branch `dev`, HEAD `4f161b4` (uv environment setup).
Codebase size: **3,132 lines** across 9 Python files. Inference only — no training code.

---

## Index

- [Session 1 — Orientation](#session-1--orientation)
  - [Original question](#original-question)
  - [1. Do all Klein models share the same architecture?](#1-do-all-klein-models-share-the-same-architecture--yes-identically)
  - [2. Data flow, first principles: prompt string -> PNG](#2-data-flow-first-principles-prompt-string--png)
  - [3. Positional encoding — the deep dive](#3-positional-encoding--the-deep-dive)
  - [4. Reading roadmap](#4-reading-roadmap)
  - [5. Architecture differences across the family](#5-architecture-differences-across-the-family--summary)
- [Session 1 — Follow-up questions](#session-1--follow-up-questions)
  - [Q1. Double blocks vs. single blocks](#q1-what-are-double-blocks-and-single-blocks)
  - [Q2. The safety filter — can I disable it?](#q2-the-safety-filter--can-i-disable-it)
  - [Q3. What is klein 9B KV?](#q3-what-is-flux2-klein-9b-kv-what-does-kv-mean)
  - [Q4. What do `prc` and `pe` stand for?](#q4-what-do-prc-and-pe-stand-for)
  - [Q5. What is modulation?](#q5-what-is-modulation)
  - [Q6. Correction — "collide at the origin"](#q6-correction--text-and-image-collide-at-the-origin)
- [Running log of useful facts](#running-log-of-useful-facts)

---

# Session 1 — Orientation

## Original question

> This is the codebase for the flux 2 family. You are running on my mac. You shouldn't run any experiment here. On the server, I have already make the model running. The model I run is flux klein base 4B, and since it is undistilled, I run it for 50 steps. The generated result looks great.
> First explore and get yourself familiar with this minimal codebase. The first question you need to help me answer is, does the models in klein family, i.e. 4B, 9B, base, and dev, follows the same architecture and other wrapper? If I developed some thing using one model, can it be easily transfered to other larger or smaller model? Such as doing some modification to the RoPE, or using it for some customized generation process? This is determine the later developing path. If it can transfer (and I suspect it can?), I will use the 4B base for initial developing, and only switch to 9B and dev later.
> Another goal of me, is to get fully understanding of the codebase. The codebase itself is not large, in the end I will want to read it line by line and understand every detail, since flux is a really important reference work of my own project. But at first, looking at thousands of lines, I don't know where to start, don't know which part is the core logic, which part is standard transformers of diffusers part, and which are just boilerplate, and which are irrelevant at first. I know flux is a rectified flow model, I am familiar with rectified flow matching, and I know it uses autoencoder for latent diffusion, and text encoder (which is pretty large??? seems like it uses a 9B qwen text encoder, what is that???) and other things special for a text-to-image model (and looks like it also support image to image, but partitioning the 128 head dim to 4 * 32, which correspond to group, H, W and text token position???). When inspecting implementation detail, I will especially want to focus on the implementaion and use of positional encoding, since my later project is going to modify and implement my own. Please give me a road map of how should I approach reading the codebase, with the goal of fully understanding line by line.
> Continue from above goal, I will want to understand how data flows in first principle. That is, I will want to track the process, of, starting with a prompt string and reference image (does all model support reference images?), how are they being transform into tokens? which model which component does this? and how does it flows through the "main" flow matching transfomer? and how does the decode part work (from latent to image by vae? this is latent diffusion). Please give me this thing along with the code reading roadmap.
> Also, what is the architecture's difference between different models in the family?

---

## 1. Do all Klein models share the same architecture? -> Yes, identically

There is exactly **one** transformer class, `Flux2` (`src/flux2/model.py:52`). All five Klein
variants *and* FLUX.2 [dev] instantiate the same class with a different `@dataclass` of
hyperparameters (`model.py:10-49`). There is no per-model subclass, no per-model wrapper, no
conditional architecture code.

| | Klein 4B | Klein 9B | FLUX.2 [dev] |
|---|---|---|---|
| `hidden_size` | 3072 | 4096 | 6144 |
| `num_heads` | 24 | 32 | 48 |
| **`head_dim`** | **128** | **128** | **128** |
| `depth` (double blocks) | 5 | 8 | 8 |
| `depth_single_blocks` | 20 | 24 | 48 |
| `axes_dim` (RoPE) | **[32,32,32,32]** | **[32,32,32,32]** | **[32,32,32,32]** |
| `theta` | **2000** | **2000** | **2000** |
| `mlp_ratio` | **3.0** | **3.0** | **3.0** |
| `in_channels` (latent) | **128** | **128** | **128** |
| `context_in_dim` | 7680 | 12288 | 15360 |
| `use_guidance_embed` | False | False | True |
| Text encoder | Qwen3-4B-FP8 | Qwen3-8B-FP8 | Mistral-Small-3.2-24B |
| Params (computed) | 3.88 B | 9.08 B | 32.2 B |

**The single most important fact for the project: `head_dim == 128` for every model, and
`axes_dim`/`theta` are byte-for-byte identical.** The constructor even asserts this
(`model.py:62-64`): `sum(axes_dim)` must equal `hidden_size // num_heads`. So `EmbedND`,
`rope()` and `apply_rope()` are *literally the same computation* on 4B, 9B and dev — the only
thing that changes is how many heads broadcast against the shared `pe` tensor.

**Verdict: develop on Klein-base-4B. A RoPE modification transfers to 9B and dev with zero code
changes.** The `[32,32,32,32]` split and `theta=2000` are not derived from model size, so
nothing structural needs re-tuning.

### The "4B / 9B" naming

Those numbers refer to the **DiT only**, not the text encoder. Parameter counts computed from
the configs: 3.88B, 9.08B, 32.2B — matching the published names. The text encoder is a
*separate*, unshipped model pulled from HF.

### Caveats when transferring (all real, all small)

1. **Three near-duplicate forward paths.** `forward()` (`model.py:115`), `forward_kv_extract()`
   (`:170`), `forward_kv_cached()` (`:267`). Same for both block classes. Editing `forward()`
   only will silently break `flux.2-klein-9b-kv`. The other five models never touch the KV
   paths.
2. **Distilled vs. base changes the *sampler*, not the model.** Klein-base-4B goes through
   `denoise_cfg` (`sampling.py:364`), which **doubles the batch**
   (`img = torch.cat([img, img])` at `:375`) for real CFG against an empty prompt. Distilled
   models use `denoise` with batch size 1. 50 steps x CFG = **100 forward passes**, not 50.
3. **`use_guidance_embed` only exists on dev.** Klein passes `guidance=1.0` into `denoise`,
   builds `guidance_vec`, and then *completely ignores* it because `self.use_guidance_embed` is
   False (`model.py:128`).
4. **`context_in_dim` differs** -> `txt_in` weight shape differs. Irrelevant for RoPE.
5. **CLI gotcha:** for any Klein model, `cli.py:288` *additionally* loads the full 24B Mistral
   just for safety filters and prompt upsampling. See [Q2](#q2-the-safety-filter--can-i-disable-it).

---

## 2. Data flow, first principles: prompt string -> PNG

Reference images: **yes, all six models support single- and multi-reference editing** (README
table; `encode_image_refs` is called model-agnostically at `cli.py:443`).

### A. Prompt string -> context tokens

```
"a photo of a forest..."
  |- Qwen3Embedder.forward                        text_encoder.py:384
     |- apply_chat_template([{role:"user", content:prompt}],
     |     add_generation_prompt=True, enable_thinking=False)   :390
     |- tokenizer(..., padding="max_length", truncation=True,
     |     max_length=512)                                      :397
     |- self.model(input_ids, output_hidden_states=True)        :411
     |- stack hidden_states at layers [9, 18, 27]               :418  <- OUTPUT_LAYERS_QWEN3
     |- rearrange "b c l d -> b l (c d)"                        :419
        -> (1, 512, 3 x d_llm)     e.g. (1, 512, 7680) for Qwen3-4B
```

Three non-obvious things:

- **It is not a sentence embedding and not the last layer.** It's three *intermediate* layers
  concatenated along the feature axis. That's why `context_in_dim = 3 x d_llm`
  (2560x3=7680, 4096x3=12288, 5120x3=15360). The DiT reads a multi-scale slice of the LLM's
  residual stream.
- **Sequence length is always exactly 512**, right-padded. The attention mask is *never
  propagated to the DiT* — pad tokens become real context tokens the image attends to. The
  model learned to ignore them.
- For base models, `cli.py:565-571` encodes `""` and the prompt separately and concatenates
  along **batch** for CFG.

Into the DiT: `txt = self.txt_in(ctx)` — one `Linear(context_in_dim -> hidden_size, bias=False)`
(`model.py:70`). That is the *entire* text adapter.

Then `batched_prc_txt(ctx)` (`cli.py:571` -> `sampling.py:93`) attaches position ids:

```python
coords = {"t": [0], "h": [0], "w": [0], "l": arange(512)}
x_ids = cartesian_prod(t, h, w, l)        # (512, 4)
```

### B. Reference images -> context tokens

```
PIL images
  |- encode_image_refs                            sampling.py:52
     |- default_prep                                       :65 -> :226
     |   |- to_rgb, cap_min_pixels (rejects AR > 8:1)
     |   |- cap_pixels: 2024^2 for 1 ref, 1024^2 for >=2    :55-60
     |   |- center_crop to multiple of 16                   :235
     |   |- 2*x - 1   -> [-1, 1]                            :223
     |- ae.encode(img)                                      :72
     |- listed_prc_img(encoded, t_coord=[10, 20, 30, ...])  :76-80
```

`ae.encode` (`autoencoder.py:314`):

```
(3, H, W) -> Encoder (3 downsamples, ch_mult [1,2,4,4]) -> (64, H/8, W/8)
          -> chunk -> mean only, discard logvar (deterministic)  :316
          -> pixel-unshuffle 2x2: "c (i pi) (j pj) -> (c pi pj) i j"  :318
          -> (128, H/16, W/16)
          -> BatchNorm2d(128, affine=False) in eval mode   :304  <- latent normalization
```

Effective compression: **16x spatial, 128 channels**. FLUX.2 replaced FLUX.1's fixed
`scale_factor`/`shift_factor` scalars with a **per-channel BatchNorm** whose running stats ship
in the checkpoint. `decode` inverts it manually (`:308-312`).

Then `prc_img` (`sampling.py:141`) flattens and assigns ids:

```python
coords = {"t": [10*(k+1)], "h": arange(h), "w": arange(w), "l": [0]}
x = rearrange(x, "c h w -> (h w) c")
```

### C. Target latent

```python
shape = (1, 128, height // 16, width // 16)                cli.py:581
randn = torch.randn(shape, generator=..., dtype=bfloat16)       :583
x, x_ids = batched_prc_img(randn)                               :584
# ids: (t=0, h, w, l=0)
```

The target latent and the text *both* live at `t=0`; references live at `t=10, 20, 30...`.

### D. The flow-matching loop

```python
timesteps = get_schedule(num_steps, image_seq_len)          cli.py:586 -> sampling.py:244
```

`get_schedule` = `linspace(1, 0, n+1)` warped by `generalized_time_snr_shift` (`:240`), with
`mu` fit empirically from `(image_seq_len, num_steps)` via a piecewise-linear formula
(`compute_empirical_mu`, `:251`). Resolution-dependent timestep shifting: more tokens -> more
time spent at high noise.

Then (`sampling.py:364`, the path Klein-base-4B takes):

```python
for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    img_input     = cat([img, img_cond_seq], dim=1)      # [target, refs] along SEQUENCE
    img_input_ids = cat([img_ids, img_cond_seq_ids], dim=1)
    pred = model(x=img_input, x_ids=..., timesteps=t_vec, ctx=txt, ctx_ids=...)
    pred = pred[:, :img.shape[1]]                        # discard ref-slot predictions
    pred_u, pred_c = pred.chunk(2)
    pred = pred_u + guidance * (pred_c - pred_u)         # CFG
    img = img + (t_prev - t_curr) * pred                 # Euler
```

Standard rectified-flow convention: `x_t = (1-t)*x_0 + t*eps`, so `v = eps - x_0`, and since
`t_prev < t_curr` the step moves toward data. **Plain Euler, no higher-order solver, no noise
re-injection.** Reference images ride along as extra sequence positions every single step —
exactly the waste the 9B-KV variant eliminates.

### E. Inside `Flux2.forward` (`model.py:115-168`)

```
vec = time_in(timestep_embedding(t, 256))                      :126-127
      (+ guidance_in(...) if dev)                              :128-130

# ---- THREE global modulation layers, computed ONCE ----
double_block_mod_img = double_stream_modulation_img(vec)       :132
double_block_mod_txt = double_stream_modulation_txt(vec)       :133
single_block_mod, _  = single_stream_modulation(vec)           :134

img = img_in(x)     # Linear(128 -> hidden)                    :136
txt = txt_in(ctx)   # Linear(context_in_dim -> hidden)         :137

pe_x   = pe_embedder(x_ids)      # (B,1,N_img,64,2,2)          :139
pe_ctx = pe_embedder(ctx_ids)    # (B,1,512,  64,2,2)          :140

for block in double_blocks:                                    :142
    img, txt = block(...)        # separate weights, JOINT attention

img = cat((txt, img), dim=1)     # now one stream               :153
pe  = cat((pe_ctx, pe_x), dim=2)                                :154

for block in single_blocks:      # 20/24/48 of them             :156
    img = block(...)

img = img[:, num_txt_tokens:]    # drop text                    :165
img = final_layer(img, vec)      # -> 128 channels = velocity   :167
```

**Big structural point:** modulation is *hoisted out of the blocks*. In FLUX.1 every block owned
its own `Modulation`. Here there are exactly three for the whole network (`model.py:98-108`),
shared by all blocks. See [Q5](#q5-what-is-modulation).

**Second structural point:** "double stream" means **two sets of weights, one joint attention**.
`_prepare_qkv` (`model.py:569`) computes txt and img q/k/v with separate projections, then
`cat`s them (`:594-596`) and runs one SDPA over `[txt, img]`. It is MMDiT, not cross-attention.

### F. Latent -> image

```python
x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)      cli.py:616 -> sampling.py:24
```

`scatter_ids` un-flattens tokens back onto a (t,h,w) grid using their position ids —
`flat_ids = t*w*h + h*w + w` and a `scatter_`. For pure t2i this is an identity-ish reshape, but
it's written generally so it also works when the sequence carries multiple grids.

```python
x = ae.decode(x)        autoencoder.py:327
    |- inv_normalize (undo BatchNorm)                 :308
    |- pixel-shuffle 2x2 -> (32, H/8, W/8)            :329
    |- Decoder (3 upsamples) -> (3, H, W)             :335
x = x.clamp(-1, 1); (127.5*(x+1)) -> PIL -> EXIF -> save    cli.py:628-639
```

---

## 3. Positional encoding — the deep dive

### The four axes

`ids` is an integer tensor `(B, N, 4)`. The axes, in order:

| idx | name | target latent | text | reference image *k* |
|---|---|---|---|---|
| 0 | `t` — group / reference slot | `0` | `0` | `10*(k+1)` |
| 1 | `h` — latent row | `0..H/16-1` | `0` | `0..h_k-1` |
| 2 | `w` — latent col | `0..W/16-1` | `0` | `0..w_k-1` |
| 3 | `l` — text token index | `0` | `0..511` | `0` |

Set in `prc_img` (`sampling.py:141-151`) and `prc_txt` (`:93-103`), with the `t` offsets at
`sampling.py:76` (`scale = 10`).

Three design consequences to internalize before modifying anything:

- **References share the `h,w` coordinate frame with the target.** A reference pixel at
  `(h=5,w=7)` has the *same* h/w RoPE phase as the target latent at `(h=5,w=7)`. Only the `t`
  axis separates them. This is what makes spatially-aligned editing (inpainting, upsampling,
  style transfer at matching positions) work almost for free — and it's the mechanism to
  preserve or deliberately break.
- **Text and image collide at the origin.** See [Q6](#q6-correction--text-and-image-collide-at-the-origin).
- **`t` gaps are 10, not 1.** With `theta=2000` and 32 dims on that axis, a gap of 10 is a large,
  unambiguous phase separation between references.

### The implementation

```python
# model.py:818
def rope(pos, dim, theta):
    scale = arange(0, dim, 2) / dim          # [0, 1/16, ..., 15/16]  for dim=32
    omega = 1.0 / (theta ** scale)           # 1.0 ... 8.06e-4
    out = einsum("...n,d->...nd", pos, omega)
    out = stack([cos, -sin, sin, cos], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)   # explicit 2x2 matrices
```

```python
# model.py:694
class EmbedND(nn.Module):
    def forward(self, ids):
        emb = cat([rope(ids[..., i], axes_dim[i], theta) for i in range(4)], dim=-3)
        return emb.unsqueeze(1)              # (B, 1, N, 64, 2, 2)  <- the 1 is the HEAD axis
```

```python
# model.py:828
def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)     # (B,H,L,64,1,2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), ...
```

Four observations that matter for designing a replacement:

1. **Explicit 2x2 matrix form, not `(cos, sin)` + `rotate_half`.** `EmbedND` materializes a real
   `(..., 64, 2, 2)` tensor and `apply_rope` does the matrix-vector product by hand. Anything
   expressible as a 2x2 linear map per frequency pair drops in with **no change to
   `apply_rope`** — non-unit-determinant scaling, shearing, a learned/predicted 2x2, decaying
   magnitudes. Unusually hospitable base for RoPE surgery. (Cost: 4 floats per pair instead of
   2, in fp32, for the whole sequence.)
2. **`unsqueeze(1)` at `model.py:707` is the head-broadcast axis.** All heads currently share one
   positional scheme. Per-head or per-group positional encoding = emit `(B, H, N, 64, 2, 2)`
   here; nothing downstream changes.
3. **`pe` is computed exactly twice per forward** (`model.py:139-140`) and passed by reference
   into every block. Two consumers: `model.py:649` (double, on `pe_full` built at `:599`) and
   `model.py:496` (single, on the `pe` concatenated at `:154`). Total surface area for a RoPE
   change: **`EmbedND.forward`, `rope`, `apply_rope`, and the two `prc_*` id-builders.**
   About 40 lines.
4. **`theta = 2000`, not 10000.** With `dim=32` per axis, `omega` runs from `1.0` down to
   `2000^(-15/16) ~= 8.1e-4`, i.e. wavelengths from `2*pi` to `~7800`. Tuned for `<=~200`-long
   spatial axes (a 1440px side = 90 latent positions), not for 100k-token LLM contexts. If the
   position range is extended, this is the knob.

---

## 4. Reading roadmap

Tiered so it's never 3,000 lines at once. **Total genuinely-novel code is ~450 lines.**

### Tier 0 — skip entirely for now (~1,100 lines)

| File / range | Why |
|---|---|
| `watermark.py` (47) | Already commented out at `cli.py:27, 618`. Dead. |
| `system_messages.py` (82) | Prompt strings for the safety filter. |
| `openrouter_api_client.py` (129) | HTTP client for optional prompt upsampling. |
| `cli.py:1-419` + `:445-561` | REPL parsing, config dataclass, OpenRouter plumbing, safety checks. |
| `text_encoder.py:33-363` | The entire Mistral embedder — only used by `flux.2-dev`. |
| `docs/*` | `flux2_dev_hf.md` is a diffusers tutorial; `flux2_klein_kv_cache.md` worth 5 min *later*. |

### Tier 1 — the core (read line by line, ~330 lines) — START HERE

**Session 1: the sequence contract (~120 lines).** Understand what a "token" is before looking
at the network.

- `sampling.py:93-156` — `prc_txt`, `prc_img`, the `listed`/`batched` wrappers. Where the
  4-axis id scheme is born.
- `sampling.py:52-90` — `encode_image_refs`, the `t`-offset trick.
- `sampling.py:12-49` — `compress_time`, `scatter_ids` (the inverse).
- `cli.py:565-584` + `:616-617` — where these are called end to end.

**Session 2: the transformer (~180 lines).**

- `model.py:10-49` — the three configs.
- `model.py:52-168` — `__init__` and `forward`. Read `forward` twice.
- `model.py:400-412` (`Modulation`), `:415-434` (`LastLayer`), `:390-397` (`SiLUActivation` =
  SwiGLU).
- `model.py:524-635` — `DoubleStreamBlock._prepare_qkv` / `_apply_residuals`. Joint attention,
  split weights.
- `model.py:437-484` — `SingleStreamBlock._qkv` / `_out`. Fused `linear1` producing qkv **and**
  the MLP in one matmul.
- `model.py:758-815` — `causal_attn_fn`. **Important:** despite the name, on the default path
  `num_ref_tokens=0`, so `q_ref`/`k_ref` are empty and it degenerates to one plain bidirectional
  SDPA over `[txt, img+refs]`. The masking only activates for 9B-KV.

**Session 3: positional encoding (~40 lines) — the focus.**

- `model.py:694-707`, `:818-833`. Then trace the two `apply_rope` call sites (`:496`, `:649`)
  and the `pe` concatenations (`:154`, `:599`).

**Session 4: the sampler (~70 lines).**

- `sampling.py:240-266` — schedule + `compute_empirical_mu`.
- `sampling.py:269-307` — `denoise`.
- `sampling.py:358-410` — `denoise_cfg` <- **the one Klein-base-4B actually runs.**

### Tier 2 — the autoencoder (~340 lines, skim 300 / read 40)

`autoencoder.py:24-268` is the **stock Stable-Diffusion / LDM VAE**, essentially verbatim:
`ResnetBlock`, `AttnBlock`, `Downsample`/`Upsample`, `Encoder`, `Decoder`. Skim to confirm
`ch_mult=[1,2,4,4]` -> 3 downsamples -> /8.

Read carefully only `autoencoder.py:271-336` — the `AutoEncoder` wrapper. That's the
FLUX.2-specific part: pixel-unshuffle to reach /16 & 128 channels, and BatchNorm latent
normalization.

### Tier 3 — the text encoder (~55 lines)

`text_encoder.py:366-436` (`Qwen3Embedder` + loaders) and the constants at `:26-28`. The
`Mistral3SmallEmbedder` above it is 330 lines of safety filtering and prompt upsampling that
Klein never touches.

### Tier 4 — KV caching (~200 lines, only if targeting 9B-KV)

`model.py:170-330`, `:329-372` (modulation blending), `:486-522`, `:637-681`, and the
`kv_cache is not None` branch of `causal_attn_fn`. Read `docs/flux2_klein_kv_cache.md` first.

### What's standard vs. what's FLUX.2-specific

**Standard / boilerplate (recognize and move on):** the entire VAE body; `timestep_embedding`
(`model.py:710`, the DDPM sinusoidal embedding); `RMSNorm`/`QKNorm` (`:734-755`); `MLPEmbedder`;
`LastLayer` (adaLN-Zero from DiT); the double/single block *skeleton* (inherited from FLUX.1).

**FLUX.2-specific, worth attention:**

- 4-axis RoPE `[32,32,32,32]`, `theta=2000`, explicit 2x2 matrices (FLUX.1 was 3-axis
  `[16,56,56]`, `theta=10000`).
- **Modulation hoisted to three global layers** instead of per-block.
- SwiGLU MLPs (`SiLUActivation`) instead of GELU; `bias=False` throughout.
- **No CLIP pooled vector.** `vec` is timestep (+guidance) only; all text conditioning arrives
  as sequence tokens.
- Text context = **3 concatenated intermediate LLM layers**, 512 fixed length, no attention mask
  forwarded.
- 128-channel latents at /16 via pixel-unshuffle, with BatchNorm-based normalization.
- Reference images as plain extra sequence tokens distinguished by a `t`-axis offset — no
  separate cross-attention, no ControlNet-style branch.

---

## 5. Architecture differences across the family — summary

**Weights differ. Architecture does not.** Precisely:

| Difference | Where | Affects the work? |
|---|---|---|
| Width / depth | `model.py:10-49` | No — `head_dim` is 128 everywhere |
| `use_guidance_embed` | dev only, `model.py:73-74` | No |
| `context_in_dim` | follows text encoder | No |
| Text encoder (Qwen3-4B / 8B / Mistral-24B) | `util.py:22,34,82` | Only if touching text conditioning |
| Sampler: `denoise` vs `denoise_cfg` vs `denoise_cached` | `cli.py:587-615` | **Yes** — batch is 2x on base models |
| Attention mask (ref-isolating) | 9B-KV only | Only if targeting 9B-KV |
| Step/guidance distillation | weights, plus `fixed_params` guard at `util.py:25` | Distilled models *refuse* non-default steps/guidance (`cli.py:212-238`) |

`flux.2-klein-base-4b` has `fixed_params: {}` (`util.py:62`), so it's the only 4B variant that
allows changing `num_steps` and `guidance` freely. The distilled ones hard-error on deviation.

**Caveat:** `encode_image_refs` hardcodes `.cuda()` at `sampling.py:72`, so reference-image
encoding is CUDA-only as written — relevant for any CPU/MPS smoke test on the Mac.

---

# Session 1 — Follow-up questions

## Q1. What are double blocks and single blocks?

Both are transformer blocks over the same token sequence. The difference is **how much the two
modalities share**, and **how attention and MLP are wired**.

### `DoubleStreamBlock` (`model.py:524-680`) — 26 d^2 params each

- Keeps `img` and `txt` as **two separate tensors**, each with its own residual stream.
- **Two complete sets of weights**: `img_norm1/img_attn/img_norm2/img_mlp` and
  `txt_norm1/txt_attn/txt_norm2/txt_mlp`. A double block is ~2x the parameters of an ordinary
  transformer block at the same width.
- **The attention is joint, not separate.** `_prepare_qkv` (`:569`) projects each modality with
  its own `qkv`, then concatenates: `q = cat((txt_q, img_q), dim=2)` (`:594-596`), and one SDPA
  runs over the whole `[txt, img]` sequence. Text sees image, image sees text, in the same
  softmax. This is MMDiT (SD3) — **not** cross-attention, **not** two isolated towers.
- **Sequential** sub-layers: attention branch, then MLP branch, each with its own gate -> needs
  6 modulation values per stream.

### `SingleStreamBlock` (`model.py:437-521`) — 13 d^2 params each

- Operates on **one** tensor holding `[txt, img]` concatenated (done once at `model.py:153`).
- **One shared set of weights** for both modalities.
- **Fused and parallel**: `linear1` maps `d -> 3d (qkv) + 6d (SwiGLU MLP input)` in a single
  matmul (`:453-457`). Attention and MLP are computed **in parallel** off the same normalized
  input, then `linear2` maps `cat(attn_out, mlp_out): (d + 3d) -> d` (`:459`). One residual, one
  gate -> needs only 3 modulation values. (Same trick as GPT-J / PaLM parallel blocks.)
- Exactly **half** the parameters of a double block at the same width.

### Why both

Lineage: SD3's MMDiT was all-double. FLUX.1 introduced the hybrid (19 double + 38 single).
FLUX.2 pushes further:

| | double : single | ratio |
|---|---|---|
| Klein 4B | 5 : 20 | 1:4 |
| Klein 9B | 8 : 24 | 1:3 |
| FLUX.2 dev | 8 : 48 | 1:6 |

Rationale: early on the two modalities have wildly different statistics (LLM hidden states vs.
VAE latents), so modality-specific weights help. After a few blocks they've been fused into a
common space, and shared weights are 2x more parameter-efficient per block — plus the fused
matmuls are more GPU-efficient.

**Worth noting:** text tokens keep getting updated through *all* the single blocks, then are
thrown away at `model.py:165`. On dev that's 48 blocks of compute on 512 text tokens whose
outputs are discarded.

---

## Q2. The safety filter — can I disable it?

**Yes, trivially. It is not in the model and not in the library — it is three lines in the CLI.**

Verified with a full-repo grep. Call sites:

| Line | What it checks |
|---|---|
| `scripts/cli.py:353` | the prompt (`test_txt`) |
| `scripts/cli.py:362` | each input image (`test_image`) |
| `scripts/cli.py:633` | the generated output (`test_image`) |

**Zero call sites anywhere in `src/flux2/`.** The `Flux2` module, the autoencoder, the samplers
and `Qwen3Embedder` never invoke it. Nothing is baked into the weights.

### What is actually loaded

`cli.py:286-290`:

```python
text_encoder = load_text_encoder(model_name, device=torch_device)
if "klein" in model_name:
    mod_and_upsampling_model = load_text_encoder("flux.2-dev")   # <- Mistral-Small-3.2-24B
else:
    mod_and_upsampling_model = text_encoder
```

So on any Klein model the CLI loads a **second, 24B model** purely for moderation + optional
prompt upsampling, plus a `Falconsai/nsfw_image_detection` classifier pipeline
(`text_encoder.py:54`). It is loaded unconditionally at startup, before any generation, even
with `upsample_prompt_mode="none"` (the default). On Klein-4B that dwarfs the 3.9B generator.

`Qwen3Embedder.test_txt` / `test_image` raise `NotImplementedError` (`text_encoder.py:421-425`),
which is why the CLI pulls in Mistral rather than reusing Qwen.

### How to disable

**Best option for research: write your own driver and skip `cli.py` entirely.** Roughly 30
lines, no filter, no Mistral, no REPL:

```python
import torch
from flux2.util import load_flow_model, load_ae, load_text_encoder
from flux2.sampling import batched_prc_img, batched_prc_txt, get_schedule, denoise_cfg, scatter_ids, encode_image_refs

name = "flux.2-klein-base-4b"
dev  = torch.device("cuda")
te   = load_text_encoder(name, device=dev).eval()      # Qwen3-4B only
net  = load_flow_model(name, device=dev).eval()
ae   = load_ae(name).eval()

prompt, H, W, steps, cfg_scale, seed = "a misty forest", 768, 1360, 50, 4.0, 0
with torch.no_grad():
    ctx = torch.cat([te([""]), te([prompt])], dim=0).to(torch.bfloat16)   # CFG: uncond + cond
    ctx, ctx_ids = batched_prc_txt(ctx)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x, x_ids = batched_prc_img(torch.randn((1,128,H//16,W//16), generator=g,
                                           dtype=torch.bfloat16, device="cuda"))
    ts = get_schedule(steps, x.shape[1])
    x  = denoise_cfg(net, x, x_ids, ctx, ctx_ids, timesteps=ts, guidance=cfg_scale)
    img = ae.decode(torch.cat(scatter_ids(x, x_ids)).squeeze(2)).float().clamp(-1, 1)
```

**Or patch the CLI:** set `mod_and_upsampling_model = None` at `cli.py:288` and guard the three
call sites (`:353`, `:362`, `:633`). Note `:633` is inside an `if/else` whose `else` branch does
the actual save — invert or remove the condition, don't just delete the `if`.

### Does it apply to `diffusers`?

The filter in *this* repo is CLI-level Python, not part of the checkpoint, so it **cannot**
follow the weights into diffusers. A `from_pretrained` pipeline loads the transformer, VAE and
text encoder — none of which contain it.

Caveats: (a) the class name in this repo's docs is `Flux2Pipeline`, not `Flux2KleinPipeline`
(`docs/flux2_dev_hf.md`) — check what your diffusers version exposes; (b) diffusers is not
installed on this Mac, so I could not verify its internals here. Verify in one line on the
server:

```python
print(pipe.components.keys())          # look for "safety_checker" / "feature_extractor"
```

Historically the FLUX pipelines in diffusers ship no safety checker (unlike the old
`StableDiffusionPipeline`), so I expect the answer is "nothing to disable" — but confirm it
rather than assume.

---

## Q3. What is FLUX.2 klein 9B KV? What does "KV" mean?

**KV = the Key and Value tensors in attention** — the same "KV cache" idea as in LLM decoding.

### The observation

Reference image tokens are *constant* across denoising steps. Their content never changes. So in
principle their K and V in every attention layer could be computed once and reused for all
subsequent steps, instead of being recomputed 4 (or 50) times.

### Why that is false in the normal model

In standard FLUX.2 the attention is fully bidirectional: ref tokens attend to the text **and to
the noisy target latent**. So after layer 1 their hidden states depend on the current noise
level -> their K/V genuinely change every step. Nothing to cache.

### What the KV variant changes

Two modifications, both of which make the ref tokens' states provably step-independent:

1. **Attention isolation** (`causal_attn_fn`, `model.py:758-815`). Reference tokens attend
   **only to other reference tokens**: `attn_ref = SDPA(q_ref, k_ref, v_ref)` (`:811`), while
   txt and img still attend to everything (`:806`). Despite the function name it isn't causal —
   it's a block mask with refs as an isolated block.
2. **Timestep isolation** (`model.py:180, 198, 210-211`). Ref tokens are modulated at a **fixed**
   `ref_fixed_timestep = 0.0` rather than the current `t`. The per-token modulation tensor is
   assembled by `_blend_double_mods` / `_blend_single_mods` (`:346-372`).

With both, the ref tokens' hidden states — and therefore their K/V — are *bit-identical* at
every step. Step 0 (`forward_kv_extract`) runs the full sequence and stores `k_ref`/`v_ref` per
block (`:501-503`, `:653-656`); steps 1+ (`forward_kv_cached`) drop ref tokens from the input
entirely and splice the cached K/V into attention (`:772-786`).

### Why it needs its own checkpoint

This **changes the forward computation**, so it is not a drop-in inference trick. The attention
restriction has to be trained in, or quality collapses. Hence a separate weights file,
`flux-2-klein-9b-kv.safetensors`, and a separate HF repo.

The modeling cost: ref tokens can no longer see the prompt or the evolving target. That's a real
sacrifice, traded for speed.

### When it helps

From `docs/flux2_klein_kv_cache.md`: **1.21x - 2.66x**, scaling with (ref tokens) / (total
tokens). More references + smaller output = bigger gain. And it only activates when refs exist —
`cli.py:588-592` requires `use_kv_cache and ref_tokens is not None`. **Pure text-to-image gets
zero benefit** (there are no ref tokens to cache).

### Why doesn't dev have it?

The repo doesn't say; this is my inference:

- It requires a **separate finetune and a separate released checkpoint** — real cost per model.
- The payoff scales with ref-token fraction *and* matters most where latency is the product.
  Klein is positioned as sub-second interactive editing, where 2x on 1.5s is the whole pitch.
  Dev is positioned as maximum quality with no latency constraint.
- The attention restriction is a **quality regression**. On a flagship 32B model that's a worse
  trade than on a speed-oriented 9B.

---

## Q4. What do `prc` and `pe` stand for?

- **`prc` = "process".** `prc_txt` / `prc_img` = process text / process image into
  `(tokens, position_ids)`. Confirmed by the wrapper names `batched_prc_img`, `listed_prc_img`
  (`sampling.py:154-156`) and `img_ctx_prep` (`:65`). Just an abbreviation, no deeper meaning.
- **`pe` = positional embedding / positional encoding.** `self.pe_embedder` (`model.py:67`) is
  the `EmbedND` module; `pe_dim` (`:62`) is the per-head dimension the PE must fill; `pe_x`,
  `pe_ctx`, `pe_full` are the computed rotation tensors.
- **Bonus:** inside `apply_rope` the argument is named `freqs_cis` (`model.py:828`) — inherited
  from Meta's Llama RoPE code, where rotations were stored as complex numbers
  (`cis(theta) = cos(theta) + i*sin(theta)`). FLUX stores real 2x2 matrices instead but kept the
  name. When a name looks odd in this codebase, it's usually a lineage artifact.

### Other abbreviations in this codebase

| Abbrev | Meaning |
|---|---|
| `ctx` | context = the text embeddings |
| `vec` | the global conditioning vector (timestep [+ guidance]) |
| `mod` | modulation (shift / scale / gate) |
| `ae` | autoencoder |
| `t_off` | time-axis offset assigned to reference images |
| `ps` | patch size for the pixel-shuffle (`autoencoder.py:295`) |
| `img_cond_seq` | reference tokens — "image conditioning, **seq**uence-wise" (concatenated along sequence, as opposed to channel-wise) |
| `sft` | safetensors (`load_sft`) |
| `_l` | sequence length (`prc_txt`) |
| `nin_shortcut` | "network-in-network" 1x1 conv shortcut — inherited from the original LDM VAE |

---

## Q5. What is modulation?

**Modulation = how the network is conditioned on the timestep.** It's the `adaLN-Zero` mechanism
from DiT (Peebles & Xie, 2022), used by every model in this lineage.

### The problem

A diffusion transformer must condition on a continuous scalar `t` (and, for dev, a guidance
value). Two options: (a) append it as an extra token, or (b) use it to modulate the network's
normalization. DiT found (b) works much better, and it became standard.

### The mechanism, in three steps

**1. Strip LayerNorm's learned parameters.**

```python
nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)   # model.py:464, 537, 545, ...
```

Normalizes, but has no learned gamma/beta.

**2. Generate gamma/beta from the condition (this is FiLM).**

```python
x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift          # model.py:470, 580
```

`shift` and `scale` come from a linear layer applied to the conditioning vector. Written as
`1 + scale` so that `scale = 0` is the identity.

**3. Gate the residual branch (this is the "-Zero").**

```python
return x + mod_gate * output                                    # model.py:484, 626
```

At initialization the gate-producing weights are ~0, so every block starts as an identity
function. Training is stable no matter how deep the network is.

### So each residual branch needs a `(shift, scale, gate)` triple

- `SingleStreamBlock` has **one** residual branch (attention and MLP run in parallel, summed by
  `linear2`, one gate) -> 3 values -> `Modulation(double=False)`, `multiplier=3`.
- `DoubleStreamBlock` has **two** branches per stream (attention, then MLP) -> 6 values ->
  `Modulation(double=True)`, `multiplier=6`. And there are two streams (img, txt) -> two such
  modules.

That's exactly `self.multiplier = 6 if double else 3` at `model.py:404`.

The generator (`model.py:400-412`):

```python
out = self.lin(silu(vec))          # (B, multiplier*d)
out = out.chunk(self.multiplier, dim=-1)
return out[:3], out[3:] if self.is_double else None
```

and `vec` is built at `model.py:126-130`: sinusoidal timestep embedding -> 2-layer MLP, plus a
guidance embedding on dev.

### The FLUX.2 surprise — remember this one

There are only **four** modulation generators in the entire network:
`double_stream_modulation_img`, `double_stream_modulation_txt`, `single_stream_modulation`
(`model.py:98-108`), and `LastLayer.adaLN_modulation` (`:424`).

They are called **once** in `forward` (`:132-134`), and the **same tuples are passed into every
block** (`:142-151`, `:156-163`). All 8 double blocks share one identical `(shift, scale, gate)`
set; all 48 single blocks share another.

In DiT and in FLUX.1 each block had its own `Modulation`. FLUX.2 made it **global**. Two
consequences:

- Large parameter saving: `15 d^2` total instead of `~(6*2*depth + 3*single) d^2`.
- **The network's entire timestep-dependence funnels through one tensor per forward.** That
  makes `vec` an extremely high-leverage hook — a very small, very central place to intervene if
  I want per-block, per-token or otherwise structured timestep behaviour.

And there's already a worked example of making it token-dependent in the codebase:
`_blend_double_mods` / `_blend_single_mods` (`model.py:346-372`) build a **per-token**
modulation tensor by splicing ref-timestep values into the shared one. Read those if I ever want
spatially- or semantically-varying modulation.

---

## Q6. Correction — "text and image collide at the origin"

**My earlier phrasing was wrong and the pushback was right.** I said the modalities are
disambiguated "by content and by the separate weight streams". The separate-weight-stream part
only holds in the **double** blocks. In the **single** blocks — 20 of 25 on Klein-4B, 48 of 56 on
dev — text and image go through **exactly the same `linear1`, `linear2`, `QKNorm`, `pre_norm`,
the same attention heads, and the same modulation values**. They share everything.

### The facts

- Text token `l=0` has ids `(0,0,0,0)`. The target latent at `(h=0,w=0)` has ids `(0,0,0,0)`.
  Identical. And at position 0 every rotation is the identity matrix (`cos 0 = 1, sin 0 = 0`),
  so RoPE does nothing to either.
- More broadly: **every** image token has `l=0`; **every** text token has `h=w=0`. The axes
  separate positions *within* a modality; they do not separate the modalities from each other.
- There is **no modality embedding, no segment embedding, no type token** anywhere in the model.

### So what actually distinguishes them

1. **Different entry projections.** `img_in: Linear(128 -> d)` and
   `txt_in: Linear(context_in_dim -> d)` are different matrices reading completely different
   input spaces (`model.py:68, 70`). The two modalities land in different regions of the
   `d`-dimensional residual stream from token 0, and nothing forces them to overlap.
2. **The double blocks amplify that separation** with modality-specific weights, before the
   streams merge at `model.py:153`.
3. **After the merge it is purely content.** The single blocks treat every token identically;
   they distinguish text from image the way a decoder-only LM distinguishes a noun from a verb —
   by where the vector points, not by a tag.

### Why this matters for my project

- **RoPE position is genuinely ambiguous across modalities in FLUX.2.** A positional scheme that
  assumes "same position => same role" will be wrong here.
- **Most of the positional capacity is idle.** Count the identity rotations:

  | token type | active axes | identity (unrotated) head dims |
  |---|---|---|
  | text | `l` only | **96 of 128** |
  | target latent | `h, w` | **64 of 128** |
  | reference latent | `t, h, w` | **32 of 128** |

  For every text token, 3 of the 4 axes sit at position 0, so 96 of 128 q/k dimensions receive
  the identity rotation. That's a large amount of unused positional bandwidth — either free
  capacity to repurpose, or evidence that the `[32,32,32,32]` split is generous by design.

---

# Running log of useful facts

Append anything worth not re-deriving.

- **Param counts** (computed from configs, no weights loaded): Klein-4B 3.88B, Klein-9B 9.08B,
  dev 32.2B. Formula per block: double = `26 d^2`, single = `13 d^2`, global modulation =
  `15 d^2`.
- **`causal_attn_fn` is a misnomer on the default path.** With `num_ref_tokens=0` it is plain
  bidirectional attention plus one empty SDPA call.
- **Klein-base at 50 steps + CFG = 100 network evaluations**, not 50 (`denoise_cfg` doubles the
  batch).
- **No `__init__.py`** in `src/flux2/` — it's an implicit namespace package, which is why the
  README runs `PYTHONPATH=src python scripts/cli.py`.
- **`encode_image_refs` hardcodes `.cuda()`** (`sampling.py:72`) — blocks any CPU/MPS path.
- **The VAE is deterministic at inference**: `encode` takes only the mean and discards the
  log-variance half (`autoencoder.py:316`), so no sampling in the latent.
- **Latent normalization is a `BatchNorm2d` with `affine=False`** whose running stats ship in
  `ae.safetensors` (`autoencoder.py:296-312`) — replacing FLUX.1's scalar
  `scale_factor`/`shift_factor`.
- **Reference image resolution caps**: 2024^2 pixels for a single reference, 1024^2 each when
  there are 2+ (`sampling.py:55-60`). Aspect ratio beyond 8:1 raises.
- **Text is always exactly 512 tokens**, right-padded, and the padding mask is never forwarded
  to the DiT.
