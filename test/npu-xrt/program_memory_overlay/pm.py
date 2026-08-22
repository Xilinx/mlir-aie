#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: dont_run
# RUN: echo PASS | FileCheck %s
# CHECK: PASS

"""Driver for the program-memory overlay tests.

Everything the RUN lines need, over pmlib. Subcommands:

  geometry   print a named layout, or check that an invalid one is refused
  workload   generate and compile a dummy overlay of an exact size
  emit       write the design MLIR (and the slot.ld fragment) for a recipe
  link       link an overlay object into a slot and verify it
  sizes      report resident + overlay footprint against program memory
  check      the resident did not move between passes
  compare    overlay output matches per-overlay references
"""

import argparse
import sys

from pmlib import design as pmdesign
from pmlib import workload as pmworkload
from pmlib.elf import find_core_elf, text_size, text_words
from pmlib.geometry import GeometryError, RECIPES, recipe
from pmlib.link import OverlayError, link as pmlink


def cmd_geometry(args):
    g = recipe(args.recipe)
    try:
        g.validate()
    except GeometryError as e:
        if args.expect_invalid:
            print(f"refused: {e}")
            return
        sys.exit(f"{args.recipe}: unexpectedly invalid: {e}")
    if args.expect_invalid:
        why = g.why_invalid or "no reason recorded"
        sys.exit(f"{args.recipe}: expected to be refused ({why}), but it validated")
    print(g.describe())


def cmd_workload(args):
    n = pmworkload.compile_overlay_of_size(
        args.tag, args.n_elems, args.output, args.size, workdir=args.workdir
    )
    print(f"overlay {args.tag}: {n} bytes")


def parse_phases(spec):
    """ "0,1,2" -> (0,1,2); "0*98" -> 98 copies of overlay 0.

    The repeat form exists because the resident's size is linear in the number
    of phase bodies (about 80 bytes each), so driving it to the slot boundary
    takes ~98 of them -- and spelling those out would make the RUN line
    unreadable. Repeating a *phase* rather than adding dead code keeps the
    design self-consistent: the core does one acquire/release per phase and the
    runtime one fill/drain, so the counts still match and the design can run.
    """
    out = []
    for part in spec.split(","):
        if "*" in part:
            idx, count = part.split("*")
            out.extend([int(idx)] * int(count))
        else:
            out.append(int(part))
    return tuple(out)


def cmd_emit(args):
    g = recipe(args.recipe)
    if args.slot_ld:
        pmdesign.emit_slot_ld(g, args.slot_ld)
    payloads = tuple(text_words(p) for p in args.payload) if args.payload else ()
    poison = text_words(args.poison) if args.poison else ()
    cfg = pmdesign.Config(
        geometry=g,
        n_elems=args.n_elems,
        phases=parse_phases(args.phases),
        payloads=payloads,
        poison=poison,
        corrupt=tuple(
            tuple(int(v) for v in c.split(":")) for c in (args.corrupt or [])
        ),
        skip_write=tuple(int(v) for v in (args.skip_write or [])),
        wrong_address=tuple(
            tuple(int(v, 0) for v in w.split(":")) for w in (args.wrong_address or [])
        ),
    )
    with open(args.out, "w") as f:
        print(pmdesign.build(cfg), file=f)


def cmd_link(args):
    g = recipe(args.recipe)
    slot = next((s for s in g.slots if s.name == args.slot), None)
    if slot is None:
        sys.exit(f"{args.recipe} has no slot named {args.slot!r}")
    try:
        pmlink(
            args.object,
            args.resident,
            slot.base,
            slot.size,
            args.output,
            col=g.tile[0],
            row=g.tile[1],
            geometry=g,
        )
    except OverlayError as e:
        if args.expect_rejected:
            print(f"refused: {e}")
            return
        sys.exit(str(e))
    if args.expect_rejected:
        sys.exit(f"{args.output}: expected to be refused, but it linked cleanly")


def cmd_sizes(args):
    g = recipe(args.recipe)
    resident = text_size(find_core_elf(args.resident, *g.tile))
    total = resident
    print(f"  {'resident':<18} {resident:6d} bytes")
    for e in args.overlays:
        n = len(text_words(e)) * 4
        total += n
        print(f"  {e.split('/')[-1]:<18} {n:6d} bytes")
    pm = g.program_memory_size
    print(f"  {'total':<18} {total:6d} bytes of {pm} ({100.0 * total / pm:.0f}%)")
    if args.require_exceeds and total <= pm:
        sys.exit(
            f"this design totals {total} bytes, which still fits in {pm}: it no "
            f"longer demonstrates a program larger than program memory"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("geometry")
    g.add_argument("recipe", choices=sorted(RECIPES))
    g.add_argument("--expect-invalid", action="store_true")
    g.set_defaults(func=cmd_geometry)

    w = sub.add_parser("workload")
    w.add_argument("--tag", type=int, required=True)
    w.add_argument("--size", type=lambda v: int(v, 0))
    w.add_argument("--n-elems", type=int, default=256)
    w.add_argument("--workdir", default=".")
    w.add_argument("--output", required=True)
    w.set_defaults(func=cmd_workload)

    e = sub.add_parser("emit")
    e.add_argument("--recipe", default="one_slot")
    e.add_argument("--slot-ld")
    e.add_argument("--payload", action="append")
    e.add_argument("--poison")
    e.add_argument("--phases", default="0")
    e.add_argument("--n-elems", type=int, default=256)
    e.add_argument("--corrupt", action="append", metavar="PHASE:WORD")
    e.add_argument("--skip-write", action="append", metavar="PHASE")
    e.add_argument("--wrong-address", action="append", metavar="PHASE:DELTA")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_emit)

    l = sub.add_parser("link")
    l.add_argument("--object", required=True, action="append")
    l.add_argument("--resident", required=True)
    l.add_argument("--recipe", default="one_slot")
    l.add_argument("--slot", default="a")
    l.add_argument("--output", required=True)
    l.add_argument("--expect-rejected", action="store_true")
    l.set_defaults(func=cmd_link)

    s = sub.add_parser("sizes")
    s.add_argument("--resident", required=True)
    s.add_argument("--recipe", default="one_slot")
    s.add_argument("--overlays", required=True, nargs="+")
    s.add_argument("--require-exceeds", action="store_true")
    s.set_defaults(func=cmd_sizes)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
