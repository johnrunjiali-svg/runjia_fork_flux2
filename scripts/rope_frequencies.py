"""Choose, plane by plane, which RoPE frequencies the "quantized" torus RoPE makes periodic.

    uv run python scripts/rope_frequencies.py                                 n = 64: 1024 px tiles of 16 px tokens
    uv run python scripts/rope_frequencies.py --n 48                          another grid length
    uv run python scripts/rope_frequencies.py --threshold 0.05                start from: omega >= 0.05 rounded, the rest kept
    uv run python scripts/rope_frequencies.py --threshold 0.05 --rule fundamental    ... or never below the fundamental
    uv run python scripts/rope_frequencies.py --threshold 0.05 --out configs/rope/fast.json --yes   write that, no questions

The table has one row per rotation plane of one axis (h and w get the same rules, each with its own
n): the trained frequency omega, how many cycles it makes across one edge, and what it becomes
under each rule of flux2.torus.quantize. A rotation is periodic on the edge exactly when its cycle
count is an integer, so the rules are ways of picking that integer:

    round        the nearest one, 0 included: a plane below half a cycle per edge goes blind
    fundamental  the nearest one but at least 1: the slowest rotation that still repeats
    keep         the trained frequency, not periodic
    an integer   that many cycles per edge
    cos, sin     the trained frequency on a position bent into a circle (torus.positions): periodic
                 without touching the frequency; meant in pairs, cos on one plane, sin on the next

At the prompt, type per-plane edits and press enter to see the table again; an empty line accepts:

    7:k          plane 7 keeps its frequency              k = keep, r = round, f = fundamental
    7-15:f       planes 7 to 15 go to the fundamental    c = cos, s = sin, cs = the two alternating
    6-15:cs      cos, sin, cos, sin, ... over 6 to 15    an integer is a cycle count: 4-6:1
    r r r r f f f k k k k k k k k k                      all sixteen at once

The choice is written as json; run_experiments.py takes it as --rope <file>. The sheet header and
index.html show it as the one-letter string, r r r f ..., in the same order as this table.
"""

import json
import math
from pathlib import Path

from flux2.torus import frequencies, quantize

WORDS = {"k": "keep", "r": "round", "f": "fundamental", "c": "cos", "s": "sin"}


def table(omega, n: int, rules: list) -> str:
    """One row per plane: the trained frequency, its cycles per edge, and what each rule makes of it."""
    all_round, all_fundamental, chosen = (quantize(omega, n, r).tolist() for r in (None, ["fundamental"] * len(omega), rules))

    def cell(w: float) -> str:
        """Cycles per edge, an integer when the plane is periodic, and the frequency."""
        cycles = w * n / (2 * math.pi)
        whole = abs(cycles - round(cycles)) < 1e-4
        return (f"{cycles:5.0f}" if whole else f"{cycles:5.2f}") + f" {w:8.4f}"

    head = "cycles/edge  omega"
    rows = [f"{'plane':>5}  {head:>14}  {head:>14}  {head:>14}  {'':>11}  {head:>14}",
            f"{'':>5}  {'trained':>14}  {'round':>14}  {'fundamental':>14}  {'rule':>11}  {'result':>14}"]
    for i, rule in enumerate(rules):
        result = f"{rule + ', on a circle':>14}" if rule in ("cos", "sin") else cell(chosen[i])
        rows.append(
            f"{i:5}  {cell(omega[i].item())}  {cell(all_round[i])}  {cell(all_fundamental[i])}  {str(rule):>11}  {result}"
        )
    kept, blind = sum(rule == "keep" for rule in rules), chosen.count(0.0)
    circle = sum(rule in ("cos", "sin") for rule in rules)
    rows.append(
        f"\n{len(rules) - kept - blind - circle} planes periodic on {n} tokens, {circle} on a circle, "
        f"{blind} blind to position, {kept} kept as trained (not periodic)"
    )
    return "\n".join(rows)


def edit(rules: list, line: str) -> list:
    """`7:k`, `7-15:f`, `4-6:1`, several per line, or all planes at once."""

    def rule(token: str):
        token = WORDS.get(token, token)
        if token in WORDS.values():
            return token
        if token.isdigit():
            return int(token)
        raise ValueError(f"{token!r} is not k, r, f or a cycle count")

    tokens = line.split()
    if len(tokens) == len(rules) and not any(":" in t for t in tokens):
        return [rule(t) for t in tokens]
    rules = list(rules)
    for token in tokens:
        planes, _, what = token.partition(":")
        assert what, f"{token!r}: want plane:rule, e.g. 7:k or 7-15:f"
        first, _, last = planes.partition("-")
        cycle = list(what) if len(what) > 1 and all(ch in WORDS for ch in what) else [what]  # cs: alternate
        for j, i in enumerate(range(int(first), int(last or first) + 1)):
            rules[i] = rule(cycle[j % len(cycle)])
    return rules


def name(rules: list) -> str:
    """r7k9 for seven rounded planes then nine kept; an explicit cycle count is n<k>."""
    out, run = [], []
    for rule in rules + [None]:
        if run and rule != run[0]:
            letter = f"n{run[0]}" if isinstance(run[0], int) else run[0][0]
            out.append(letter + (str(len(run)) if len(run) > 1 else ""))
            run = []
        run.append(rule)
    return "".join(out)


def main(
    n: int = 64,  # tokens along the edge: pixels / 16
    dim: int = 32,  # head dims per axis: dim / 2 planes
    theta: int = 2000,
    threshold: float = 0.0,  # planes with omega >= threshold start on `rule`, the rest on keep
    rule: str = "round",  # round or fundamental: what the planes above the threshold start on
    out: str | None = None,  # default configs/rope/<name>.json
    yes: bool = False,  # write the starting choice without asking
):
    omega = frequencies(dim, theta)
    rules = [rule if w >= threshold else "keep" for w in omega.tolist()]
    print(table(omega, n, rules))
    while not yes:
        line = input("\nedits (empty line to accept): ").strip()
        if not line:
            break
        try:
            rules = edit(rules, line)
        except (ValueError, AssertionError, IndexError) as e:
            print(e)
            continue
        print(table(omega, n, rules))

    path = Path(out or f"configs/rope/{name(rules)}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rules": rules,  # the only key run_experiments.py reads; the rest records this table
                "n": n,
                "theta": theta,
                "dim": dim,
                "omega": [round(w, 6) for w in omega.tolist()],
                "quantized": [round(w, 6) for w in quantize(omega, n, rules).tolist()],
            },
            indent=2,
        )
    )
    print(f"\nwrote {path}\n  uv run python scripts/run_experiments.py --experiments 3 --rope {path}")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
