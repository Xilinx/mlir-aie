# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Just enough ELF32 to place and inspect overlays.

Promoted to `aie.iron.overlay._elf` (the same logic backs the production
`iron.overlay` API's build pipeline); this module re-exports it so pmlib and
`pm.py` need no separate copy.
"""

from aie.iron.overlay._elf import (
    CTOR_SECTIONS,
    SHF_ALLOC,
    SHN_ABS,
    SHN_UNDEF,
    SHT_SYMTAB,
    OverlayELFError,
    defined_symbols,
    find_core_elf,
    max_stack_frame,
    peano,
    read_elf,
    section_file_offset,
    stack_frames,
    text_size,
    text_words,
    undefined_symbols,
)

__all__ = [
    "CTOR_SECTIONS",
    "SHF_ALLOC",
    "SHN_ABS",
    "SHN_UNDEF",
    "SHT_SYMTAB",
    "OverlayELFError",
    "defined_symbols",
    "find_core_elf",
    "max_stack_frame",
    "peano",
    "read_elf",
    "section_file_offset",
    "stack_frames",
    "text_size",
    "text_words",
    "undefined_symbols",
]
