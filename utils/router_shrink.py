#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Delta-debug a design that shows a router problem down to a minimal repro.

    python utils/router_shrink.py case.mlir --match "reaches .* twice"
        [--hops-off] [--out repros/name.mlir]

The property holds when routing the design fails with stderr matching, or
succeeds with a verifier problem matching, the --match regex. Each step
removes a flow, a source or destination, a program, a core, the runtime
sequence, a pre-placed switchbox op, or an attribute, or moves to a smaller
device, and keeps the change when the property still holds.
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "test" / "create-packet-flows")
)
import router_properties as rp  # noqa: E402


def holds_fn(pattern, hops_on, verbose=False, timeout=60):
    rx = re.compile(pattern)
    calls = [0]

    def holds(d):
        calls[0] += 1
        try:
            text = d.emit()
        except (KeyError, IndexError, ValueError):
            return False
        t0 = time.monotonic()
        p = rp.aie_opt(text, hops_on, timeout=timeout)
        spent = time.monotonic() - t0
        if verbose and spent > 1:
            print(f"  slow aie-opt: {spent:.1f}s", file=sys.stderr)
        if p is None or p.returncode < 0:
            return p is not None and bool(rx.search("crash " + p.stderr))
        if p.returncode:
            return bool(rx.search(p.stderr))
        try:
            again = rp.load_design(text)
            problems, _ = rp.verify(again, rp.Analysis(again), p.stdout, hops_on)
        except Exception as e:  # a model limit, not the property
            if verbose:
                print(f"  model: {e!r}", file=sys.stderr)
            return False
        return any(rx.search(x) for x in problems)

    holds.calls = calls
    return holds


def candidates(d):
    """Smaller designs, roughly biggest cut first."""
    for attr in ("sequences", "cores", "programs", "boxes", "muxes"):
        if getattr(d, attr):
            e = d.copy()
            if attr == "programs":
                e.programs, e.sequences, e.allocs = [], [], {}
            elif attr == "sequences":
                e.sequences, e.allocs = [], {}
                e.programs = [p for p in e.programs if p["kind"] == "start"]
            else:
                setattr(e, attr, {})
            yield e
    for k in range(len(d.packet_flows)):
        e = d.copy()
        del e.packet_flows[k]
        yield e
    for k in range(len(d.flows)):
        e = d.copy()
        del e.flows[k]
        yield e
    for k, f in enumerate(d.packet_flows):
        for key in ("srcs", "dsts"):
            for i in range(len(f[key]) if len(f[key]) > 1 else 0):
                e = d.copy()
                del e.packet_flows[k][key][i]
                yield e
    for k, p in enumerate(d.programs):
        if p["kind"] == "start":
            e = d.copy()
            del e.programs[k]
            e.sequences = [
                [
                    (
                        (ev[0], ev[1] - (ev[1] > k), *ev[2:])
                        if ev[0] in ("start", "await")
                        else ev
                    )
                    for ev in s
                ]
                for s in e.sequences
            ]
            yield e
    for tile in list(d.cores):
        e = d.copy()
        del e.cores[tile]
        yield e
    for tile, ops in d.boxes.items():
        for i in range(len(ops)):
            e = d.copy()
            del e.boxes[tile][i]
            yield e
    for k, f in enumerate(d.packet_flows):
        for key in ("keep", "priority", "mask"):
            if f[key] is not None:
                e = d.copy()
                e.packet_flows[k][key] = None
                yield e
    used = {n for ops in d.cores.values() for _, n, _ in ops}
    used |= {op[2] for p in d.programs for b in p["seq"] for op in b if op[0] == "lock"}
    unused = [n for n in d.locks if n not in used]
    if unused:
        e = d.copy()
        for n in unused:
            del e.locks[n]
        yield e
    if d.tiles:
        e = d.copy()
        e.tiles = []
        yield e
    fam = [x for devs in rp.FAMILIES.values() for x in devs if d.dev in devs]
    for dev in fam[: fam.index(d.dev)] if d.dev in fam else []:
        e = d.copy()
        e.dev = dev
        try:
            e.used_tiles()
            if all(e.target.exists(t) for t in e.used_tiles()):
                yield e
        except (KeyError, AttributeError):
            pass


def shrink(d, holds, log=None):
    changed = True
    while changed:
        changed = False
        for e in candidates(d):
            if holds(e):
                d, changed = e, True
                if log:
                    log(d)
                break
    return d


def size(d):
    return (
        len(d.flows)
        + sum(len(f["srcs"]) + len(f["dsts"]) for f in d.packet_flows)
        + len(d.programs)
        + len(d.cores)
        + sum(len(v) for v in d.boxes.values())
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("design", type=Path)
    cli.add_argument("--match", required=True)
    cli.add_argument("--hops-off", action="store_true")
    cli.add_argument("--out", type=Path)
    cli.add_argument("-v", action="store_true")
    args = cli.parse_args()
    d = rp.load_design(args.design.read_text())
    holds = holds_fn(args.match, not args.hops_off, args.v)
    if not holds(d):
        sys.exit("the property does not hold on the input")

    def log(d):
        if args.v:
            print(f"  size {size(d)} after {holds.calls[0]} runs", file=sys.stderr)

    d = shrink(d, holds, log)
    text = d.emit()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
