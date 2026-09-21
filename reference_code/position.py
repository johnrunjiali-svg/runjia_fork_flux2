"""Where a token is, and how attention is told.

Self-attention is permutation-equivariant: shuffle the tokens and the output
shuffles with them, unchanged. A convolution never has this problem -- the
kernel is laid over the grid, so pixel ``(3, 7)`` neighbours ``(3, 8)`` by
construction -- but a transformer knows nothing about the grid unless position
is put in by hand. For a *denoiser* the failure is unusually visible, because
the output is an image: a network that cannot tell the top-left patch from the
bottom-right one produces the right local texture in an arrangement that is not
a picture of anything.

There are exactly three places the information can enter an attention layer,
and the interface here is those three places::

    tokens    x_i  <- x_i + P(p_i)                     an absolute code per position
    q, k      q_i  <- R(p_i) q_i,  k_j <- R(p_j) k_j   a rotation per position
    logits    s_ij <- s_ij + b(p_j - p_i)              a bias per displacement

:class:`PositionEncoding` declares all three as no-ops and every scheme below
overrides exactly one, so the choice is a config string and nothing downstream
changes. The transformer calls :meth:`~PositionEncoding.prepare` once per
forward -- position depends on the grid, not on the batch or the layer -- and
hands the result to every block.

**Rotary, in two dimensions.** Attention reads its input only through the inner
product ``q_i . k_j``. So rotate both by an orthogonal, position-dependent
``R(p)`` and the logit becomes

    (R(p_i) q_i)^T (R(p_j) k_j) = q_i^T (R(p_i)^T R(p_j)) k_j

-- the entire positional content of the layer is the one matrix
``R(p_i)^T R(p_j)``. Pick ``R`` so that this depends only on the displacement
``p_j - p_i`` and the layer is *relative* by construction: for every pair of
tokens, every head, and every input.

The construction. Split a head's ``d`` coordinates into ``d/2`` planes, and
rotate plane ``m`` by an angle that is **linear** in position::

    alpha_m(p) = omega_m . p,     omega_m in R^2
    R(p)       = blockdiag(rot(alpha_1(p)), ..., rot(alpha_{d/2}(p)))

Linearity is the whole trick. Rotations of a plane compose by *adding* angles,
so ``rot(a)^T rot(a') = rot(a' - a)``, and

    R(p)^T R(p') = blockdiag(rot(omega_m . (p' - p)))

which depends on the two positions only through their difference -- and depends
on that difference only through the ``d/2`` numbers ``omega_m . (p' - p)``. Note
what falls out for free: the origin cancels, so where the grid is numbered from
cannot matter, and every layer is translation-invariant whether or not the
training data is.

**Choosing the frequencies.** Write ``omega_m = phi_m (cos theta_m,
sin theta_m)``, which separates the two decisions a 2D frequency has to make:
``theta`` is the *direction* in the image along which the plane measures
displacement, and ``phi`` is *how fast* its phase turns per token of it. Then

    alpha_m(p' ) - alpha_m(p) = phi_m * <(cos theta_m, sin theta_m), p' - p>

is the displacement projected onto direction ``theta_m``, scaled by ``phi_m``.
Two directions at ``0`` and ``pi/2`` give the familiar axial 2D RoPE (half the
planes see row displacement, half see column); more directions let a plane
respond to diagonal structure without composing two planes to do it.

The magnitude is best read as a **wavelength in tokens**, ``lambda = 2 pi /
phi``: the displacement over which the plane's phase comes all the way back
round. Both ends of the ladder are bounded by the grid:

- Below ``lambda = 2`` tokens the phase turns more than half a circle per token,
  so distinct displacements alias onto the same rotation immediately. That is
  the Nyquist limit, and it is the default ``shortest_wavelength``.
- Above ``lambda = 2 * grid`` the phase turns less than half a circle across the
  *whole image*, so the plane is nearly constant and is spending two of the
  head's coordinates saying almost nothing.

This is worth spelling out because the number everyone inherits from 1D text
RoPE, ``base = 10000``, puts the slowest wavelength at ``2 pi * 10000 ~ 63000``
tokens. On an 8x8 grid of patches that is four orders of magnitude past useful,
and most of the head does nothing. It is the same failure as the aliasing note
in :mod:`diffusion.models.embeddings`, approached from the other end: a
frequency bank is only right relative to the range its argument actually covers.

One further constraint on ``omega`` is worth knowing about, because it turns the
band from a judgement call into a finite list: ask the rotation to be unchanged
when the displacement *wraps* around the grid and ``omega`` is forced onto the
dual lattice ``2 pi (a / gh, b / gw)``, ``a, b`` integers -- the DFT basis of the
token torus. That is :class:`RopeTorus2D`, for generating images that are
seamless four ways.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..utils.registry import Registry

POSITIONS: Registry["PositionEncoding"] = Registry("position")


def _token_grid(gh: int, gw: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """``[gh * gw, 2]`` of ``(row, col)``, row-major -- the order tokens are flattened in.

    Numbered from zero. For the rotary schemes the origin cancels out of every
    logit (see above), so this is a free choice; for the absolute ones it is the
    convention the learned table is indexed by.
    """
    rows = torch.arange(gh, device=device, dtype=dtype)
    cols = torch.arange(gw, device=device, dtype=dtype)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=-1)


def _wavelengths(n: int, shortest: float, longest: float) -> torch.Tensor:
    """``n`` wavelengths in token units, geometric from ``shortest`` to ``longest``.

    Geometric rather than linear because what a scale ladder should cover evenly
    is *octaves*: the interesting structure between 2 and 4 tokens is as much as
    that between 16 and 32.

    A ladder of one rung is the geometric mean of the two ends, not
    ``torch.linspace``'s left endpoint -- a head narrow enough to afford a single
    wavelength per direction should spend it in the middle of the useful band,
    not at the aliasing limit.
    """
    if shortest <= 0 or longest < shortest:
        raise ValueError(f"need 0 < shortest <= longest, got {shortest}, {longest}")
    if n < 1:
        raise ValueError(f"need at least one wavelength, got {n}")
    if n == 1:
        return torch.tensor([math.sqrt(shortest * longest)])
    return torch.exp(torch.linspace(math.log(shortest), math.log(longest), n))


def _torus_harmonics(gh: int, gw: int) -> torch.Tensor:
    """The characters of ``Z_gh x Z_gw``: integer pairs ``(a, b)``, one per +/- class.

    These are the *only* frequencies a rotary plane can have and still wrap (see
    :class:`RopeTorus2D`), so choosing a frequency band on a torus is not a
    continuous design problem -- it is picking a subset of a finite list.

    Ordered by ``|omega|`` (lowest frequency first, matching how image energy is
    distributed) with the two fundamentals ``(1, 0)`` and ``(0, 1)`` promoted to
    the front, because those two are what make the bank injective: ``2 pi dr /
    gh`` is one-to-one on ``Z_gh`` by itself, so once both are present every
    distinct displacement has a distinct rotation and everything after is
    resolution rather than range.

    Two reductions shrink the list from ``gh * gw`` to roughly half:

    - ``(0, 0)`` is the constant character: no rotation, no information.
    - ``(a, b)`` and ``(-a, -b)`` give transposed rotations, and a plane's
      contribution is ``A cos(omega . d) + B sin(omega . d)`` with ``A, B`` set
      by the content -- so negating ``omega`` only flips a sign the q/k
      projections can absorb. Keeping both would be two coordinates doing one
      plane's work.
    """

    def balanced(v: int, n: int) -> int:
        """``v in [0, n)`` mapped to the representative nearest zero."""
        return v - n if 2 * v > n else v

    seen, entries = set(), []
    for a in range(gh):
        for b in range(gw):
            if a == 0 and b == 0:
                continue
            h = (balanced(a, gh), balanced(b, gw))
            neg = (balanced((-a) % gh, gh), balanced((-b) % gw, gw))
            key = min(h, neg)                       # h == neg at the Nyquist harmonic
            if key in seen:
                continue
            seen.add(key)
            fundamental = h in ((1, 0), (0, 1))
            entries.append((0 if fundamental else 1, (h[0] / gh) ** 2 + (h[1] / gw) ** 2, h))
    entries.sort()
    return torch.tensor([h for _, _, h in entries], dtype=torch.long)


def _rotate_planes(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each adjacent pair of coordinates of ``t`` by its plane's angle.

    ``t`` is ``[B, heads, N, head_dim]`` and ``cos``/``sin`` are
    ``[N, head_dim/2]``, one angle per token per plane, broadcast over batch and
    head -- position is a property of the token, not of either.

    Coordinates are paired *adjacently* (``(0,1), (2,3), ...``), as in the
    original RoPE. The half-split pairing used by some implementations is the
    same map up to a permutation of the head's coordinates, which the preceding
    linear layer can absorb; adjacent pairing is the one that reads like the
    block-diagonal matrix above.
    """
    a, b = t.reshape(*t.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


class PositionEncoding(nn.Module):
    """Three hooks, all identity. A scheme overrides exactly one of them.

    The transformer calls :meth:`prepare` once per forward and passes the result
    (whatever it is -- a table, a pair of cos/sin tensors, ``None``) back into
    the other three. Nothing else in the model knows which scheme is in force.
    """

    def prepare(self, gh: int, gw: int, device, dtype) -> Any:
        """Everything this scheme needs for a ``gh x gw`` token grid."""
        return None

    def add_to_tokens(self, tokens: torch.Tensor, state: Any) -> torch.Tensor:
        """``[B, N, dim] -> [B, N, dim]``, before the qkv projection."""
        return tokens

    def rotate(self, q: torch.Tensor, k: torch.Tensor, state: Any):
        """``[B, heads, N, head_dim]`` each, inside the attention product."""
        return q, k

    def logit_bias(self, state: Any) -> Optional[torch.Tensor]:
        """``[1, heads, N, N]`` added to the attention logits, or ``None``."""
        return None


@POSITIONS.register("none")
class NoPosition(PositionEncoding):
    """No position at all: the null baseline, and worth running once.

    The model is then a set function on patches. It can still denoise -- most of
    the job at small ``sigma`` is local and a patch carries its own content --
    so the loss curve looks *plausible*, and the samples are scrambled. That gap
    is the honest measure of what a positional scheme is buying.
    """

    def __init__(self, dim: int, num_heads: int, grid: Sequence[int]) -> None:
        super().__init__()


@POSITIONS.register("rope_2d")
class Rope2D(PositionEncoding):
    """2D rotary embedding: ``R(p)^T R(p')`` depends only on ``p' - p``.

    The frequencies are a product grid of ``num_directions`` directions and
    ``head_dim / (2 * num_directions)`` wavelengths, so every direction gets the
    full scale ladder rather than the ladder being correlated with the angle.

    Args:
        dim: model width. Only used to derive ``head_dim = dim / num_heads``.
        num_heads: attention heads. Every head gets the same frequencies; heads
            differ through their q/k projections, which is enough.
        grid: ``(gh, gw)`` token grid, used only for the default longest
            wavelength.
        num_directions: how many directions ``theta`` the planes are spread
            over, evenly across the half-circle ``[0, pi)`` (a direction and its
            negative measure the same displacements). ``2`` is axial RoPE --
            rows and columns, the standard choice; ``4`` adds the diagonals.
            ``1`` puts every plane on the rows and is blind to columns: an
            ablation, never a default.
        shortest_wavelength: fastest plane, in tokens. Below 2 it aliases.
        longest_wavelength: slowest plane, in tokens. Defaults to twice the
            larger grid side -- half a turn across the whole image, which is as
            slow as a plane can be and still say something.
        learnable: let the optimizer move the frequency *vectors* ``omega``
            (RoPE-Mixed, Heo et al. 2024). Note that they move freely in R^2:
            the relative property above holds for any ``omega`` whatsoever, so
            training cannot break it -- it is a property of the construction,
            not of the values.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid: Sequence[int],
        num_directions: int = 2,
        shortest_wavelength: float = 2.0,
        longest_wavelength: Optional[float] = None,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        head_dim = dim // num_heads
        if head_dim * num_heads != dim:
            raise ValueError(f"num_heads {num_heads} must divide dim {dim}")
        if head_dim % 2:
            raise ValueError(
                f"rotary planes are two-dimensional, so head_dim = dim/num_heads "
                f"must be even, got {head_dim}"
            )
        planes = head_dim // 2
        if num_directions < 1 or planes % num_directions:
            raise ValueError(
                f"num_directions must divide the {planes} rotary planes "
                f"(head_dim/2), got {num_directions}"
            )
        longest = float(2 * max(grid) if longest_wavelength is None else longest_wavelength)

        # omega[m] = phi_m * (cos theta_m, sin theta_m): the frequency *vector*
        # of plane m. The angle at position p is the plain inner product
        # omega_m . p, which is what makes the difference property exact.
        lam = _wavelengths(planes // num_directions, shortest_wavelength, longest)
        theta = math.pi * torch.arange(num_directions, dtype=torch.float32) / num_directions
        direction = torch.stack([theta.cos(), theta.sin()], dim=-1)          # [J, 2]
        omega = (2 * math.pi / lam)[None, :, None] * direction[:, None, :]   # [J, M, 2]
        omega = omega.reshape(planes, 2)

        self.head_dim = head_dim
        self.num_directions = num_directions
        if learnable:
            self.omega = nn.Parameter(omega)
        else:
            # A buffer, not a parameter: fixed by the config, but moved by .to()
            # and restored with the rest of the module.
            self.register_buffer("omega", omega)

    def prepare(self, gh: int, gw: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        p = _token_grid(gh, gw, device, self.omega.dtype)   # [N, 2]
        angle = p @ self.omega.t()                          # [N, planes]
        # Computed in the frequencies' dtype (float32) and cast afterwards: a
        # phase is an absolute quantity and half precision at a wavelength of
        # two tokens is a real loss of resolution.
        return angle.cos().to(dtype), angle.sin().to(dtype)

    def rotate(self, q: torch.Tensor, k: torch.Tensor, state):
        cos, sin = state
        return _rotate_planes(q, cos, sin), _rotate_planes(k, cos, sin)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"head_dim={self.head_dim}, num_directions={self.num_directions}"


@POSITIONS.register("rope_torus_2d")
class RopeTorus2D(PositionEncoding):
    """2D rotary embedding on a torus: the rotation depends on ``p' - p`` **mod the grid**.

    For generating images that are seamless four ways -- painting on a torus, a
    tiling texture -- where the top row is as adjacent to the bottom row as any
    other pair of neighbours, and the model should not be able to tell which
    cut of the image it was handed.

    **The constraint pins the frequencies almost completely.** From
    :class:`Rope2D`, a plane's contribution to the logit is
    ``rot(omega . (p' - p))``. Ask for that to be unchanged when the
    displacement wraps -- ``d -> d + (gh, 0)`` and ``d -> d + (0, gw)`` -- and
    since a rotation is unchanged exactly by a whole turn, the requirement is

        omega . (gh, 0) = 2 pi a       and      omega . (0, gw) = 2 pi b

    for *integers* ``a, b``. So

        omega = 2 pi (a / gh, b / gw),      a, b integers

    -- the frequency vectors must lie on the **dual lattice** of the token grid,
    which is to say the planes are exactly the characters of ``Z_gh x Z_gw``:
    the 2D DFT basis. Nothing else wraps, and everything on that lattice does.

    That changes the nature of the design. In :class:`Rope2D` the frequencies
    are continuous and the wavelength band is a judgement call, argued from
    Nyquist at one end and the size of the image at the other. Here both bounds
    are *automatic*: the shortest wavelength on the lattice is 2 tokens (the
    harmonic ``gh/2``) and the longest is ``gh`` (the fundamental, one period
    across the torus -- not ``2 * gh``, because on a torus a half-period is not
    a thing). There is no band to choose. There is only a finite list of
    ``~(gh * gw) / 2`` distinct planes, and the question is which
    ``head_dim / 2`` of them to take.

    **The choice made here** is the lowest-frequency ones, in order, with the
    two fundamentals first -- see :func:`_torus_harmonics`. The fundamentals
    make the bank injective on their own; after that, low frequencies first is
    the natural-image prior (energy falls off with frequency) and it is what a
    low-pass corner of the DFT basis looks like. The honest cost: with
    ``head_dim / 2`` well under half the lattice the bank stops short of the
    finest scales -- 32 planes on a 16x16 grid reach a shortest wavelength of
    ~3.6 tokens rather than 2 -- so a wider head buys sharper relative
    resolution here in a way it does not on the continuous version.

    **What this buys, exactly.** Every other operation in
    :mod:`diffusion.models.transformer` is per-token, so making the attention
    logits a function of displacement-mod-grid makes the *whole network* exactly
    equivariant to cyclic shifts of the token grid: roll the input by a whole
    number of patches and the output rolls with it, to floating point. That is
    the property a torus needs, and it is worth being precise about its
    granularity: the shift has to be a multiple of ``patch_size``, because the
    patch embedding and the output projection are ordinary local maps that know
    nothing about wrapping. The prior is toroidal at patch resolution; sub-patch
    seams are up to the data.

    Which is the other thing to know before running it: this scheme removes
    *all* absolute position, including "how far from the edge". On CIFAR that is
    a loss -- sky really is at the top -- and the model should be expected to do
    worse than :class:`Rope2D`. It is right when the data itself is toroidal
    (textures, wrapped random crops, tiling patterns) and wrong otherwise, and
    that is a statement about the data, not about the scheme.

    Not learnable, unlike :class:`Rope2D`: periodicity *is* the constraint
    ``omega in (2 pi / gh, 2 pi / gw) Z^2``, and gradient descent does not
    respect a lattice. Moving ``omega`` off it by any amount at all breaks the
    wrap exactly.

    Args:
        dim: model width. Only used to derive ``head_dim = dim / num_heads``.
        num_heads: attention heads.
        grid: ``(gh, gw)`` token grid -- and here it is load-bearing, not a
            default: the grid *is* the torus, so it sets the lattice.
    """

    def __init__(self, dim: int, num_heads: int, grid: Sequence[int]) -> None:
        super().__init__()
        head_dim = dim // num_heads
        if head_dim * num_heads != dim:
            raise ValueError(f"num_heads {num_heads} must divide dim {dim}")
        if head_dim % 2:
            raise ValueError(
                f"rotary planes are two-dimensional, so head_dim = dim/num_heads "
                f"must be even, got {head_dim}"
            )
        planes = head_dim // 2
        gh, gw = int(grid[0]), int(grid[1])
        self.grid = (gh, gw)

        harmonics = _torus_harmonics(gh, gw)
        if len(harmonics) < planes:
            # Not a tunable: a torus this size *has* no more distinct wrapping
            # planes. Widening the head past it would duplicate frequencies.
            raise ValueError(
                f"a {gh}x{gw} token torus has only {len(harmonics)} distinct rotary "
                f"planes, but head_dim {head_dim} asks for {planes}. Raise num_heads, "
                f"lower width, or use a smaller patch_size for a finer grid."
            )
        harmonics = harmonics[:planes]
        omega = 2 * math.pi * harmonics.to(torch.float32) / torch.tensor([gh, gw], dtype=torch.float32)

        self.head_dim = head_dim
        # The integers are the design, so they are what the checkpoint carries.
        # omega is the same thing in the units the docstring argues in, kept for
        # reading and derived rather than stored.
        self.register_buffer("harmonics", harmonics)
        self.register_buffer("omega", omega, persistent=False)

    def prepare(self, gh: int, gw: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if (gh, gw) != self.grid:
            raise ValueError(
                f"the torus lattice was built for a {self.grid} token grid, got "
                f"{(gh, gw)} -- the grid is the torus, so it cannot change"
            )
        p = _token_grid(gh, gw, device, torch.long)          # [N, 2] integers
        a, b = self.harmonics[:, 0], self.harmonics[:, 1]     # [planes]
        # angle = omega . p = 2 pi (a r / gh + b c / gw), with each term reduced
        # into a single turn *before* the multiply. Written this way the wrap is
        # the visible `%` rather than an implication of where omega sits, and the
        # phase never runs to tens of radians before meeting a cosine -- which is
        # the only place float32 could blunt an otherwise exact identity.
        # Accumulated in omega's dtype -- float32 by default, and whatever the
        # module was cast to otherwise, exactly as Rope2D does. A phase is an
        # absolute quantity; half precision at a two-token wavelength is a real
        # loss of resolution, so it is never the token dtype that decides.
        real = self.omega.dtype
        frac = (p[:, :1] * a % gh).to(real) / gh + (p[:, 1:] * b % gw).to(real) / gw
        angle = 2 * math.pi * frac
        return angle.cos().to(dtype), angle.sin().to(dtype)

    def rotate(self, q: torch.Tensor, k: torch.Tensor, state):
        cos, sin = state
        return _rotate_planes(q, cos, sin), _rotate_planes(k, cos, sin)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        lam = 2 * math.pi / self.omega.norm(dim=-1)
        return (
            f"grid={self.grid}, planes={len(self.omega)}, "
            f"wavelengths {float(lam.min()):.2f}..{float(lam.max()):.2f} tokens"
        )


@POSITIONS.register("sincos_2d")
class SinCos2D(PositionEncoding):
    """The fixed absolute code: sines and cosines of row and column (ViT/MAE/DiT).

    Half the width encodes the row and half the column, each as a Fourier
    feature bank over the same wavelength ladder the rotary scheme uses -- so
    the two differ in *where* they act (on the token, once, before any layer)
    rather than in what they are made of.

    Being absolute is the whole difference. The network can learn "this is the
    left edge", which rotary cannot express; but it must learn relativity from
    data, and at test time it has never seen a position it was not trained at.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid: Sequence[int],
        shortest_wavelength: float = 2.0,
        longest_wavelength: Optional[float] = None,
    ) -> None:
        super().__init__()
        if dim % 4:
            raise ValueError(
                f"dim must be a multiple of 4 -- half the width per axis, split "
                f"into sines and cosines -- got {dim}"
            )
        longest = float(2 * max(grid) if longest_wavelength is None else longest_wavelength)
        lam = _wavelengths(dim // 4, shortest_wavelength, longest)
        self.register_buffer("freqs", 2 * math.pi / lam)

    def prepare(self, gh: int, gw: int, device, dtype) -> torch.Tensor:
        p = _token_grid(gh, gw, device, self.freqs.dtype)          # [N, 2]
        args = (p[:, :, None] * self.freqs).reshape(gh * gw, -1)   # [N, dim/2]
        return torch.cat([args.sin(), args.cos()], dim=-1).to(dtype)

    def add_to_tokens(self, tokens: torch.Tensor, state) -> torch.Tensor:
        return tokens + state


@POSITIONS.register("learned")
class LearnedPosition(PositionEncoding):
    """One free vector per grid cell, added to the token (the ViT/DiT default).

    The most expressive absolute scheme and the least structured: it can encode
    anything about a position, and it knows nothing until it has seen that
    position in training. Fixed to one grid size by construction, which is why
    it raises rather than interpolating -- silently resizing a learned table is
    a change of meaning, not a resize.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid: Sequence[int],
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.grid = (int(grid[0]), int(grid[1]))
        self.table = nn.Parameter(torch.randn(self.grid[0] * self.grid[1], dim) * init_std)

    def prepare(self, gh: int, gw: int, device, dtype) -> torch.Tensor:
        if (gh, gw) != self.grid:
            raise ValueError(
                f"learned positions were trained for a {self.grid} token grid, "
                f"got {(gh, gw)}"
            )
        return self.table.to(dtype)

    def add_to_tokens(self, tokens: torch.Tensor, state) -> torch.Tensor:
        return tokens + state


@POSITIONS.register("rel_bias_2d")
class RelativeBias2D(PositionEncoding):
    """One learned number per displacement per head, added to the logits (Swin/T5).

    The other way to be relative, and the useful contrast with rotary: this bias
    is a function of ``p_j - p_i`` *alone*, so it says "attend nearby" (or
    "attend up") regardless of what is in the tokens, while rotary makes the
    content-dependent term itself relative. They compose, and are worth
    measuring separately before deciding to.

    Zero-initialized, so the model starts as uniform attention over positions
    and grows a locality prior only if the data asks for one. Costs one table of
    ``heads x (2gh-1)(2gw-1)`` and, unlike rotary, a materialized ``[N, N]``
    logit mask.
    """

    def __init__(self, dim: int, num_heads: int, grid: Sequence[int]) -> None:
        super().__init__()
        gh, gw = int(grid[0]), int(grid[1])
        self.grid = (gh, gw)
        self.table = nn.Parameter(torch.zeros(num_heads, (2 * gh - 1) * (2 * gw - 1)))
        # Displacements run over [-(g-1), g-1] in each axis; shifting by g-1
        # makes that a flat index into the table, computed once here.
        p = _token_grid(gh, gw, dtype=torch.long)
        d = p[None, :, :] - p[:, None, :]                                  # [N, N, 2]
        self.register_buffer("index", (d[..., 0] + gh - 1) * (2 * gw - 1) + d[..., 1] + gw - 1)

    def prepare(self, gh: int, gw: int, device, dtype) -> torch.Tensor:
        if (gh, gw) != self.grid:
            raise ValueError(
                f"the relative-bias table was built for a {self.grid} token grid, "
                f"got {(gh, gw)}"
            )
        return self.table[:, self.index].unsqueeze(0).to(dtype)            # [1, heads, N, N]

    def logit_bias(self, state) -> torch.Tensor:
        return state


def build_position(name: str, *, dim: int, num_heads: int, grid: Sequence[int], **params):
    """Every scheme takes the same three geometry facts, then its own parameters."""
    return POSITIONS.build(name, dim=dim, num_heads=num_heads, grid=tuple(grid), **params)
