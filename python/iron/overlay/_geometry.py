# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Where an overlay slot may be placed in program memory, and why.

Every rule the overlay mechanism depends on is stated once, here, and checked by
`Geometry.validate()`. Ported from
test/npu-xrt/program_memory_overlay/pmlib/geometry.py -- the named-recipe table
that lived alongside it there is test-only (it names layouts by hand for
regression coverage); `iron.overlay.ProgramMemorySlot` computes a `Geometry`
from what the caller declares instead of asking for one directly.

The rule that is easy to get wrong is the write granule. A configuration write
to the granule the core is currently fetching from is silently discarded about
half the time, while a write to any other granule always lands -- measured in
test/npu-xrt/pm_write_while_running. So a slot must not share a granule with
anything that executes while the slot is written. That is a *hardware*
property with no safe default: `get_program_memory_write_granule()` returns
None where it has not been characterised, and this module refuses rather than
guessing.
"""

from dataclasses import dataclass

from ...dialects.aie import AIEDevice, get_target_model  # pyright: ignore[reportAttributeAccessIssue]

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
                f"on was measured on npu2 only (see test/npu-xrt/pm_write_while_running). "
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
