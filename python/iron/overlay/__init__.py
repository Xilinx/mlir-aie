# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Program-memory overlays: code swapped into a compute tile's program memory at run time.

An AIE core holds a fixed amount of code, with no I-cache and no spill. A
design whose kernels do not collectively fit has to split across tiles,
shrink, or reconfigure the whole device -- or write code into a core's
program memory at run time, so which code a core runs is decided while it
runs. See `test/npu-xrt/program_memory_overlay/README.md` for the mechanism
in full; this package is the IRON-idiomatic API on top of it.

- [`ProgramMemorySlot`][iron.overlay.ProgramMemorySlot] — a reserved region of
  a core's program memory, callable inside `core_fn` like a `Kernel`.
- [`ProgramMemoryOverlay`][iron.overlay.ProgramMemoryOverlay] — one piece of
  code that can be loaded into a slot.
- [`ProgramMemoryOverlayDesign`][iron.overlay.ProgramMemoryOverlayDesign] —
  runs the two-pass build a design with slots needs; a user calls
  `.compile()` once.
"""

from .design import ProgramMemoryOverlayDesign, ProgramMemoryOverlayDesignError
from .overlay import ProgramMemoryOverlay
from .slot import ProgramMemorySlot, ProgramMemorySlotError

__all__ = [
    "ProgramMemorySlot",
    "ProgramMemorySlotError",
    "ProgramMemoryOverlay",
    "ProgramMemoryOverlayDesign",
    "ProgramMemoryOverlayDesignError",
]
