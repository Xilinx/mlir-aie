# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Where an overlay slot may be placed in program memory, and why.

Every rule the overlay mechanism depends on is stated once, here, and checked by
`Geometry.validate()`. Tests exercise the validator rather than each re-deriving
the arithmetic, so a rule can only be wrong in one place.

The rule that is easy to get wrong is the write granule. A configuration write
to the granule the core is currently fetching from is silently discarded about
half the time, while a write to any other granule always lands -- measured in
../../pm_write_while_running. So a slot must not share a granule with anything
that executes while the slot is written. That is a *hardware* property with no
safe default: `get_program_memory_write_granule()` returns None where it has not
been characterised, and this module refuses rather than guessing.
"""

from dataclasses import dataclass, field

from aie.dialects.aie import AIEDevice, get_target_model

# Program memory is 128 bits wide, so it is written a line at a time.
PROG_MEM_LINE = 16


class GeometryError(Exception):
    """A slot layout that the hardware cannot honour."""


@dataclass(frozen=True)
class Slot:
    """One region of program memory that overlays are written into.

    `core_in` is the address the core is executing from while this slot is
    written, and it is what makes the write safe. For a single-slot design that
    is the resident's wait loop at 0. For ping-pong it is the *other* slot's
    granule -- which is the whole reason ping-pong needs a dispatch stub in both
    granules, since a core that returned to a resident at 0 would be executing
    the very granule being written.

    Stating it per-slot rather than assuming "the resident at 0 always executes"
    is what lets the same rule cover both layouts.
    """

    name: str
    base: int
    size: int
    core_in: int = 0

    @property
    def end(self):
        return self.base + self.size

    def __str__(self):
        return f"{self.name}[0x{self.base:x}..0x{self.end:x})"


@dataclass(frozen=True)
class Geometry:
    """A program-memory layout: what is resident, and where the slots are.

    `resident_budget` is the ceiling the generated slot.ld ASSERT enforces on the
    resident's .text. It is part of the geometry rather than a detail of the
    design, because whether a slot is safe depends on what the resident occupies.
    """

    dev: str
    tile: tuple
    slots: tuple
    resident_budget: int
    # A region written once during setup and never again: the park the core
    # waits in while the *other* granule is written. It is an overlay in every
    # mechanical sense, but its lifetime is the whole run, so it is not one of
    # the slots and nothing schedules a payload into it.
    #
    # Ping-pong needs it because the core returns from a slot into the resident,
    # which lives in granule 0 -- so on the phase where granule 0 is written the
    # core would otherwise be executing the granule being written.
    bootstrap: object = None
    # Set when a layout is deliberately invalid, so a test can say what it
    # expects to be rejected for.
    why_invalid: str = ""

    @property
    def target_model(self):
        return get_target_model(getattr(AIEDevice, self.dev))

    @property
    def program_memory_size(self):
        return self.target_model.get_program_memory_size()

    @property
    def host_offset(self):
        return self.target_model.get_program_memory_host_offset()

    @property
    def write_granule(self):
        return self.target_model.get_program_memory_write_granule()

    def granule_of(self, addr):
        g = self.write_granule
        return None if g is None else addr // g

    def validate(self):
        """Raise GeometryError unless every slot is safe to write at run time."""
        pm = self.program_memory_size
        granule = self.write_granule

        if granule is None:
            raise GeometryError(
                f"{self.dev} has no characterised program-memory write granule, "
                f"so there is no way to know which region is safe to write while "
                f"the core runs. The half-granule behaviour this mechanism relies "
                f"on was measured on npu2 only (see ../../pm_write_while_running). "
                f"Characterise it for {self.dev} before placing overlays there."
            )

        if self.resident_budget % PROG_MEM_LINE:
            raise GeometryError(
                f"resident budget {self.resident_budget} is not a multiple of the "
                f"{PROG_MEM_LINE}-byte program-memory line"
            )

        for s in self.slots:
            if s.base % PROG_MEM_LINE:
                raise GeometryError(
                    f"{s} does not start on a {PROG_MEM_LINE}-byte program-memory "
                    f"line, so a write to it would straddle lines"
                )
            if s.size % PROG_MEM_LINE:
                raise GeometryError(
                    f"{s} is {s.size} bytes, not a multiple of the "
                    f"{PROG_MEM_LINE}-byte program-memory line"
                )
            if s.size <= 0:
                raise GeometryError(f"{s} is empty")
            if s.end > pm:
                raise GeometryError(
                    f"{s} runs past the end of program memory (0x{pm:x})"
                )
            if s.base < self.resident_budget:
                raise GeometryError(
                    f"{s} starts below the resident budget "
                    f"(0x{self.resident_budget:x}), so the resident and the slot "
                    f"would overlap"
                )
            # The hardware rule. A slot spanning two granules cannot be written
            # safely whichever granule the core is in.
            if self.granule_of(s.base) != self.granule_of(s.end - 1):
                raise GeometryError(
                    f"{s} straddles a 0x{granule:x}-byte write granule boundary. "
                    f"A write to the granule the core is fetching from is "
                    f"silently dropped about half the time, so no part of a slot "
                    f"may share a granule with executing code."
                )
            if self.granule_of(s.base) == self.granule_of(s.core_in):
                raise GeometryError(
                    f"{s} shares a 0x{granule:x}-byte write granule with the code "
                    f"the core executes while it is written (0x{s.core_in:x}). "
                    f"About half of those writes would be silently dropped."
                )

        if self.bootstrap is not None:
            b = self.bootstrap
            if b.base % PROG_MEM_LINE or b.size % PROG_MEM_LINE:
                raise GeometryError(
                    f"bootstrap {b} is not aligned to the {PROG_MEM_LINE}-byte "
                    f"program-memory line"
                )
            if b.end > pm:
                raise GeometryError(
                    f"bootstrap {b} runs past the end of program memory " f"(0x{pm:x})"
                )
            for sl in self.slots:
                if b.base < sl.end and sl.base < b.end:
                    raise GeometryError(
                        f"bootstrap {b} overlaps {sl}. It is written once and "
                        f"then executed for the rest of the run, so a slot "
                        f"payload landing on it would overwrite the only place "
                        f"the core can wait."
                    )

        for i, a in enumerate(self.slots):
            for b in self.slots[i + 1 :]:
                if a.base < b.end and b.base < a.end:
                    raise GeometryError(f"{a} overlaps {b}")

        return self

    def describe(self):
        """One line per fact, for a test to FileCheck against."""
        g = self.write_granule
        out = [
            f"device {self.dev}",
            f"tile ({self.tile[0]},{self.tile[1]})",
            f"program-memory 0x{self.program_memory_size:x}",
            f"host-offset 0x{self.host_offset:x}",
            f"write-granule {('0x%x' % g) if g else 'unknown'}",
            f"resident-budget 0x{self.resident_budget:x}",
        ]
        out += [f"slot {s.name} 0x{s.base:x} 0x{s.size:x}" for s in self.slots]
        if self.bootstrap is not None:
            b = self.bootstrap
            out.append(f"bootstrap {b.name} 0x{b.base:x} 0x{b.size:x}")
        return "\n".join(out)


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
