"""Position encoding for Qwen-Image 2.1 on a torus: two ways to make RoPE periodic, selected by config.

Qwen-Image 2.1 rotates every head of 128 dims as 64 complex planes split over three axes,
(frame | height | width) = (8 | 28 | 28) planes, each plane m of an axis turning by omega_m * p for the
token's coordinate p on that axis, omega_m = theta^(-m / 28) with theta = 10000 (frame: theta^(-m / 8)).
Text tokens advance one shared coordinate on all three axes; the target image's tokens all share one
frame coordinate f (the text length plus the earlier images' extents) and sit on a height/width grid
*centred on zero*: h in [-(gh - gh//2), gh//2), w likewise. See docs/qwen_image_2_1.md.

The logit between query p and key q is sum over axes of x_p^T R(q - p) x_q, so position enters only
through the displacement. Two ways to make the image part periodic, and both are implemented here:

- `mode="nearest"` keeps every frequency and changes the *displacement*: on a circle of n tokens the
  displacement from p to q is its nearest periodic copy, d_near = ((q - p + n/2) mod n) - n/2. Which
  copy of a key is wanted depends on the query, so attention has to be written out by hand
  (attention.py). This is the method of src/flux2/torus.py, with the grid origin moved to the centre.
- `mode="periodic"` keeps the displacement and changes the *frequencies*: a plane whose wavelength
  divides the grid wraps exactly, so every plane that makes at least `min_cycles` full turns across
  the image is snapped to the nearest integer number of turns, k = round(omega n / 2 pi), omega' =
  2 pi k / n. Planes that turn less than that -- the low frequencies, which never complete a period
  inside the image -- are left as trained (`low_freq="keep"`), or stretched to exactly one turn
  (`"fundamental"`), or dropped (`"zero"`). The rotation stays per token, so stock fused attention
  can be used, and `scope` says whether only the target image's tokens get the new table or every
  token does. Of the 28 planes of an axis, about 8 make a full turn on a 64-token side (1024 px)
  and 10 on a 128-token side; the rest are sub-cycle.
- `mode="both"` is the nearest-copy rule on top of the snapped table.

`unanchor_text` is the same choice as in torus.py: by default an image token at (h, w) sees the text
at displacement (p - h, p - w), which tells it where it is on the grid; with it on, every image token
sees the text the way the centre token (0, 0) does, and nothing in the network can tell where the
image was cut. Condition images (references) are never touched: they are flat pictures and their
pairs with the target keep the stock displacement, which is what aligns an edit with its reference.
"""

import math
from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor, nn

AXES = ("f", "h", "w")  # frame, height, width: the order of Qwen's axes_dims_rope


@dataclass
class PeConfig:
    """Everything an experiment can change about position. `from_dict` ignores unknown keys so a web
    request or a json file can carry extra fields."""

    mode: str = "nearest"  # nearest | periodic | both | none
    wrap_h: bool = True
    wrap_w: bool = True
    unanchor_text: bool = False
    # periodic mode: where high meets low, and what to do on either side
    min_cycles: float = 1.0  # planes turning fewer times than this across the image are "low"
    max_rel_error: float | None = (
        None  # snap a plane only if |k - round k| / k is at most this; None = always
    )
    rounding: str = "nearest"  # nearest | floor | ceil, applied to k = omega n / 2 pi
    low_freq: str = "keep"  # keep | fundamental | zero
    scope: str = "image"  # image: only the target image's tokens get the snapped table; all: every token
    # execution
    q_chunk: int = 512  # rows of logits held at once in the hand-written attention

    @classmethod
    def from_dict(cls, d: dict | None) -> "PeConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self):
        assert self.mode in ("nearest", "periodic", "both", "none"), self.mode
        assert self.rounding in ("nearest", "floor", "ceil"), self.rounding
        assert self.low_freq in ("keep", "fundamental", "zero"), self.low_freq
        assert self.scope in ("image", "all"), self.scope
        if self.max_rel_error is not None and self.max_rel_error < 0:
            raise ValueError("max_rel_error must be >= 0 or None")

    @property
    def wrap(self) -> tuple[bool, bool]:
        return (self.wrap_h, self.wrap_w)

    @property
    def nearest(self) -> bool:
        return self.mode in ("nearest", "both") and any(self.wrap)

    @property
    def periodic(self) -> bool:
        return self.mode in ("periodic", "both") and any(self.wrap)

    @property
    def needs_manual_attention(self) -> bool:
        """The nearest-copy rule picks a key copy per query, and unanchoring rotates the query one way for
        text and another for images: neither fits a fused kernel. Everything else does."""
        return self.nearest or self.unanchor_text


def inv_freq(dim: int, theta: float) -> Tensor:
    """omega_m for the dim/2 planes of one axis, exactly as QwenImage21Rope.rope_params builds them."""
    return 1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float64) / dim)


def cycles(inv: Tensor, n: int) -> Tensor:
    """How many full turns each plane makes across n tokens: k = omega n / 2 pi."""
    return inv * n / (2 * math.pi)


def periodic_inv_freq(inv: Tensor, n: int, cfg: PeConfig) -> Tensor:
    """The snapped table of one axis on a circle of n tokens. Every plane that comes back here wraps
    exactly: omega' = 2 pi k / n for an integer k (0 means no rotation at all)."""
    k = cycles(inv, n)
    snap = k >= cfg.min_cycles
    rounded = {"nearest": torch.round, "floor": torch.floor, "ceil": torch.ceil}[cfg.rounding](k)
    if cfg.max_rel_error is not None:
        snap &= (k - rounded).abs() / k.clamp(min=1e-12) <= cfg.max_rel_error
    high = rounded * (2 * math.pi / n)
    low = {"keep": inv, "fundamental": torch.full_like(inv, 2 * math.pi / n), "zero": torch.zeros_like(inv)}[
        cfg.low_freq
    ]
    return torch.where(snap, high, low)


def rotations(pos: Tensor, inv: Tensor) -> Tensor:
    """[N, M] complex64: exp(i omega_m p) for every position and plane. The angle is formed in float64
    (a snapped plane relies on omega n being an exact multiple of 2 pi; text positions run to a few
    hundred) and only the unit complex number is rounded."""
    angle = pos.to(torch.float64)[:, None] * inv.to(torch.float64)[None, :]
    return torch.polar(torch.ones_like(angle), angle).to(torch.complex64)


def grid_coordinates(gh: int, gw: int) -> tuple[Tensor, Tensor]:
    """The (h, w) coordinate of every token of a gh x gw image block, row-major, centred on zero as
    QwenImage21Rope lays them out: h in [-(gh - gh//2), gh//2), w in [-(gw - gw//2), gw//2)."""
    h = torch.arange(-(gh - gh // 2), gh // 2)
    w = torch.arange(-(gw - gw // 2), gw // 2)
    return h.repeat_interleave(gw), w.repeat(gh)


def token_positions(
    img_shapes: list[tuple[int, int, int]], image_pad_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """(frame, h, w) integer coordinates of every token of the joint sequence -- QwenImage21Rope.forward
    without the table lookup. Text tokens take one shared running position on all three axes; each
    image block freezes the frame axis at the position reached and advances it by max(height, width)."""
    is_image = image_pad_mask.tolist()
    frame, h, w = [], [], []
    cursor = position = 0
    for _, height, width in img_shapes:
        start = is_image.index(True, cursor)
        text = list(range(position, position + start - cursor))
        frame += text
        h += text
        w += text
        position += len(text)
        cursor = start + height * width
        gh, gw = grid_coordinates(height, width)
        frame += [position] * (height * width)
        h += gh.tolist()
        w += gw.tolist()
        position += max(height, width)
    tail = list(range(position, position + len(is_image) - cursor))
    return (torch.tensor(frame + tail), torch.tensor(h + tail), torch.tensor(w + tail))


class TorusRope(nn.Module):
    """Drop-in for the transformer's `pos_embed`: the stock table, with the height and width planes
    snapped where `cfg` asks. In `mode="nearest"` and `mode="none"` it returns exactly what the stock
    module would, so it can stay installed."""

    def __init__(self, stock):
        super().__init__()
        self.theta = stock.theta
        self.axes_dim = list(stock.axes_dim)
        self.cfg = PeConfig(mode="none")

    def tables(self, grid: tuple[int, int]) -> dict[str, Tensor]:
        """The effective omega table per axis for the target image on `grid`: snapped on a wrapped axis
        in periodic mode, stock otherwise."""
        inv = {a: inv_freq(d, self.theta) for a, d in zip(AXES, self.axes_dim)}
        if self.cfg.periodic:
            for axis, n, do_wrap in (("h", grid[0], self.cfg.wrap_h), ("w", grid[1], self.cfg.wrap_w)):
                if do_wrap:
                    inv[axis] = periodic_inv_freq(inv[axis], n, self.cfg)
        return inv

    def forward(self, img_shapes, image_pad_mask: Tensor, device) -> Tensor:
        frame, h, w = token_positions(img_shapes, image_pad_mask.cpu())
        stock = {a: inv_freq(d, self.theta) for a, d in zip(AXES, self.axes_dim)}
        target = self.tables(tuple(img_shapes[-1][1:]))
        n_t = math.prod(img_shapes[-1])
        parts = [rotations(frame, stock["f"])]
        for axis, pos in (("h", h), ("w", w)):
            if self.cfg.scope == "all":
                parts.append(rotations(pos, target[axis]))
            else:
                rot = rotations(pos, stock[axis])
                rot[-n_t:] = rotations(pos[-n_t:], target[axis])
                parts.append(rot)
        return torch.cat(parts, dim=-1).to(device)


@dataclass
class TorusGeometry:
    """What the hand-written attention needs about the target image, built once per generation. The
    frame planes are not here: they depend on the prompt length and come with the per-branch rotary
    table at call time (attention.py)."""

    grid: tuple[int, int]
    num_target: int
    planes: dict[str, slice]  # complex planes of each axis within the 64
    dims: dict[str, slice]  # real head dims of each axis within the 128
    pe_copy_hw: Tensor  # [N_t, 56] complex: the h and w planes of every token at its other periodic copy
    use_copy_h: Tensor | None  # [N_t, N_t] bool, [query, key]: this pair wants the key's h copy
    use_copy_w: Tensor | None
    cfg: PeConfig

    @property
    def any_copy(self) -> bool:
        return self.use_copy_h is not None or self.use_copy_w is not None

    def to(self, device) -> "TorusGeometry":
        move = lambda t: None if t is None else t.to(device)  # noqa: E731
        return TorusGeometry(
            self.grid, self.num_target, self.planes, self.dims,
            move(self.pe_copy_hw), move(self.use_copy_h), move(self.use_copy_w), self.cfg,
        )  # fmt: skip


def build_geometry(rope: TorusRope, cfg: PeConfig, grid: tuple[int, int], device=None) -> TorusGeometry:
    """`grid` = (gh, gw) tokens of the target image. The nearest-copy rule on one axis of n coordinates
    (torus.py, unchanged): d_near is one of d - n, d, d + n, so R(d_near) = R(p)^T R(q + s n); a key in
    the lower half of the axis is only ever wanted at q or q + n, one in the upper half at q or q - n."""
    rope.cfg = cfg
    gh, gw = grid
    edges = [sum(d // 2 for d in rope.axes_dim[:i]) for i in range(4)]
    planes = {a: slice(s, e) for a, s, e in zip(AXES, edges, edges[1:])}
    dims = {a: slice(2 * s.start, 2 * s.stop) for a, s in planes.items()}
    tables = rope.tables(grid)
    coords = dict(zip(("h", "w"), grid_coordinates(gh, gw)))

    pe_copy, use_copy = [], {}
    for axis, n, do_wrap in (("h", gh, cfg.wrap_h), ("w", gw, cfg.wrap_w)):
        pos = coords[axis]
        r = pos - pos.min()  # 0..n-1 along this axis
        if do_wrap and cfg.nearest:
            d = r[None, :] - r[:, None]  # [query, key]: key - query
            d_near = (d + n // 2) % n - n // 2
            copy = torch.where(2 * r < n, r + n, r - n)
            use = d_near != d
            assert torch.equal(torch.where(use, copy[None, :] - r[:, None], d), d_near)
            use_copy[axis] = use
            pe_copy.append(rotations(pos + (copy - r), tables[axis]))
        else:
            use_copy[axis] = None
            pe_copy.append(rotations(pos, tables[axis]))
    geo = TorusGeometry(
        grid=grid,
        num_target=gh * gw,
        planes=planes,
        dims=dims,
        pe_copy_hw=torch.cat(pe_copy, dim=-1),
        use_copy_h=use_copy["h"],
        use_copy_w=use_copy["w"],
        cfg=cfg,
    )
    return geo.to(device) if device is not None else geo


def describe_tables(rope: TorusRope, cfg: PeConfig, grid: tuple[int, int]) -> str:
    """One line per plane of the h and w axes: the trained number of turns across the image and what
    the config turns it into. For reading before an experiment, and for the web page's hint."""
    rope.cfg = cfg
    lines = []
    for axis, n in zip(("h", "w"), grid):
        stock = inv_freq(rope.axes_dim[AXES.index(axis)], rope.theta)
        eff = rope.tables(grid)[axis]
        k0, k1 = cycles(stock, n), cycles(eff, n)
        snapped = int(((k1 - k0).abs() > 1e-9).sum())
        lines.append(f"{axis} ({n} tokens): {snapped} of {len(k0)} planes changed")
        lines += [
            f"  plane {m:2d}: {a:8.3f} turns -> {b:8.3f}"
            for m, (a, b) in enumerate(zip(k0.tolist(), k1.tolist()))
        ]
    return "\n".join(lines)
