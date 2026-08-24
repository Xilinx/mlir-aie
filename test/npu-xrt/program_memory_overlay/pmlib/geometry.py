# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Where an overlay slot may be placed in program memory, and why.

`Slot`/`Geometry`/`GeometryError`/`PROG_MEM_LINE` are promoted to
`aie.iron.overlay._geometry` (the production `iron.overlay` API computes a
`Geometry` the same way, rather than asking for one directly) and re-exported
here. The named-recipe table below is test-only -- it exists so a RUN line
can say "one_slot" or "pingpong" rather than repeating six numbers -- and has
no production equivalent.
"""

from aie.iron.overlay._geometry import (
    PROG_MEM_LINE,
    Geometry,
    GeometryError,
    Slot,
)

__all__ = [
    "PROG_MEM_LINE",
    "Geometry",
    "GeometryError",
    "Slot",
    "recipe",
    "RECIPES",
]


def _npu2(slots, resident_budget=0x2000, tile=(0, 2), **kw):
    return Geometry(
        dev="npu2", tile=tile, slots=tuple(slots), resident_budget=resident_budget, **kw
    )


# Named layouts, so a RUN line says which configuration it is testing rather
# than carrying six numbers a reviewer has to re-derive. Keeping them in one
# table also makes an inconsistency between them visible.
RECIPES = {
    # The single-slot layout: resident in the low granule, one slot in the high
    # one. Everything the mechanism does today.
    "one_slot": _npu2([Slot("a", 0x2000, 0x2000)]),
    # Same, but the resident is allowed to grow right up to the slot. Used to
    # measure the guard band and to drive the slot.ld ASSERT boundary.
    "one_slot_tight": _npu2([Slot("a", 0x2000, 0x2000)], resident_budget=0x2000),
    # Proves the blockwrite's column/row are really parameterized.
    "other_tile": _npu2([Slot("a", 0x2000, 0x2000)], tile=(1, 3)),
    # Ping-pong. Program memory is 0x4000 with a 0x2000 granule, so there are
    # exactly two granules and the two slots must take one each: while either is
    # written the core executes from the other. That is only sound once a
    # dispatch stub exists in both granules -- a core that returned to the
    # resident at 0 would be executing the granule being written. The geometry
    # is checkable now; the stub needs a mechanism that does not exist yet.
    "pingpong": _npu2(
        [
            # Listed first, and that is load-bearing: phase 0 uses the first
            # slot, and the core has to park in the resident for it. A first
            # phase that parked in the bootstrap would jump there before setup
            # had written anything into it.
            #
            # Granule 1. Written while the core waits in the resident.
            Slot("b", 0x2000, 0x1C00, core_in=0x0000),
            # Granule 0, alongside the resident. Written while the core waits in
            # the bootstrap park, which is in granule 1.
            Slot("a", 0x0400, 0x1C00, core_in=0x3C00),
        ],
        resident_budget=0x0400,
        bootstrap=Slot("stub", 0x3C00, 0x0400),
    ),
    # Deliberately invalid, one rule each.
    "bad_unaligned": _npu2(
        [Slot("a", 0x2008, 0x1000)],
        why_invalid="slot base is not on a program-memory line",
    ),
    "bad_past_end": _npu2(
        [Slot("a", 0x3000, 0x2000)],
        why_invalid="slot runs past the end of program memory",
    ),
    "bad_straddles_granule": _npu2(
        [Slot("a", 0x1000, 0x2000)],
        resident_budget=0x1000,
        why_invalid="slot straddles the write-granule boundary",
    ),
    "bad_shares_executing_granule": _npu2(
        [Slot("a", 0x0800, 0x0800, core_in=0x0)],
        resident_budget=0x0800,
        why_invalid="slot shares a write granule with the executing resident",
    ),
    "bad_overlap": _npu2(
        [Slot("a", 0x2000, 0x1800), Slot("b", 0x3000, 0x1000)],
        why_invalid="slots overlap",
    ),
    # npu1 has no characterised granule, so placing an overlay there is refused
    # rather than guessed at.
    "bad_uncharacterised_device": Geometry(
        dev="npu1",
        tile=(0, 2),
        slots=(Slot("a", 0x2000, 0x2000),),
        resident_budget=0x2000,
        why_invalid="device has no characterised write granule",
    ),
}


def recipe(name):
    if name not in RECIPES:
        raise SystemExit(
            f"unknown recipe {name!r}; known: {', '.join(sorted(RECIPES))}"
        )
    return RECIPES[name]
