"""Four-way seamless images from FLUX.2, copying only at the input: wrap the latent, run the stock model, chop.

The state of the sampler is the gh x gw token grid, the core. Before every forward pass it is
continued past each edge with a band of `band` circular copies, exactly as one pads an image with
mode="wrap", and the model is handed that (gh + 2 band) x (gw + 2 band) grid as if it were an
ordinary, larger picture. After the pass the velocity is chopped back to the core, the Euler step
moves the core, and the next step pads the *updated* core again:

    core --pad--> wide --model--> v_wide --chop--> v --Euler--> core'      (once per step)

    2 x 2 core [[A, B], [C, D]], band 1: the model sees the 4 x 4 grid on positions {-1, 0, 1, 2}^2

        D C D C
        B A B A          16 tokens in, the velocity of the middle 4 kept; the other 12 are
        D C D C          thrown away and made again from the new A B C D at the next step
        B A B A

The model is not touched: stock attention, stock RoPE, stock forward. That is the point of this
baseline. Every token of the wide grid sits at its own position, the core at 0 .. g - 1 and the
copies at -band .. -1 and g .. g + band - 1, the same positions the copy tokens of
flux2/copy_tokens.py give their copies. So the two methods put the same tokens at the same places
and differ in one thing only, *where the copying happens*:

- copy_tokens: a copy is its source's key and value, re-made from the source in every attention
  layer, and only the core asks questions.
- copy_latent: a copy is a token of the input, copied once per step. From the first layer on it is
  an ordinary token with a hidden state of its own, which drifts from its source's, and it asks
  questions of its own. Text tokens attend to it like to any image token. Its output is discarded,
  and the core never hears of the drift except through attention.

What to expect, so that the result can be read: a core token at the edge has the copies as
neighbours, so there is no cut for it to be consistent across; but the outermost copies have
nothing beyond them, are poorly conditioned, and their hidden states feed the edge tokens' attention
layer after layer. A wider band keeps those further away and costs (gh + 2 band)(gw + 2 band)
tokens of attention.

band = 0 is the stock sampler to the bit. The AE decoder is the one from torus.py, handed the latent
continued past its edges, so that its zero padding does not put the seam back: band 0 is therefore
stock sampling + circular decode, and the first column of a sweep over band.

Non-forced choices, flagged: the timestep schedule is the one of the core's token count (it depends
on the sequence length through mu), not of the wide grid, so that all columns of a sweep walk the
same schedule from the same noise; and no reference tokens, because this baseline is text to image.
"""

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from .model import Flux2


def padded_ids(grid: tuple[int, int], band: int, device) -> Tensor:
    """[N, 4] (t, h, w, l) of the (gh + 2 band) x (gw + 2 band) grid, row-major like prc_img lists a
    grid. The core keeps the positions it has without any padding."""
    gh, gw = grid
    coords = (torch.arange(1), torch.arange(-band, gh + band), torch.arange(-band, gw + band), torch.arange(1))
    return torch.cartesian_prod(*coords).to(device)


def denoise_copy_latent(
    model: Flux2,
    core: Tensor,  # [P, C, gh, gw] noise, one latent per prompt
    txt: Tensor,  # [P, N_txt, D] the prompts; at guidance != 1, [2P, N_txt, D]: P empty prompts, then the prompts
    txt_ids: Tensor,  # [P or 2P, N_txt, 4]
    timesteps: list[float],
    band: int,
    guidance: float = 1.0,
) -> Tensor:
    """Euler flow matching, `sampling.denoise` with the pad before the model and the chop after it.
    guidance = 1 is the distilled klein recipe, one pass per step; any other value is classifier-free
    guidance against the empty prompt, two passes. Returns the core, [P, C, gh, gw]."""
    gh, gw = core.shape[-2:]
    ids = padded_ids((gh, gw), band, core.device)[None]
    branches = txt.shape[0] // core.shape[0]
    assert branches == (1 if guidance == 1 else 2), f"{branches} text blocks for guidance {guidance}"
    for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:])), desc="denoise"):
        t_vec = torch.full((txt.shape[0],), t_curr, dtype=core.dtype, device=core.device)
        wide = F.pad(core, (band, band, band, band), mode="circular")  # copy ...
        x = rearrange(wide, "b c h w -> b (h w) c").repeat(branches, 1, 1)
        v = model(x, ids.expand(x.shape[0], -1, -1), t_vec, txt, txt_ids, None)
        v = rearrange(v, "b (h w) c -> b c h w", h=gh + 2 * band)
        v = v[..., band : band + gh, band : band + gw]  # ... chop
        if branches == 2:
            v_uncond, v_cond = v.chunk(2)
            v = v_uncond + guidance * (v_cond - v_uncond)
        core = core + (t_prev - t_curr) * v
    return core
