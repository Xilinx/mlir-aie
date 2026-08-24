# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Link a kernel into a program-memory slot, and refuse anything unusable.

An overlay is an ordinary object linked at the slot's address instead of at 0,
against the resident image's defined symbols so it can call resident functions
and reach resident buffers by name.

Everything `verify` rejects has the same character: it produces a program that
builds and loads and then behaves oddly, rather than a build failure. That is
the only reason these checks exist, and it is why each message says what the
consequence would have been rather than just naming the rule.

Ported from test/npu-xrt/program_memory_overlay/pmlib/link.py.
"""

import subprocess

from ._elf import (
    CTOR_SECTIONS,
    SHF_ALLOC,
    defined_symbols,
    find_core_elf,
    peano,
    read_elf,
    undefined_symbols,
)
from ._geometry import PROG_MEM_LINE, GeometryError


class OverlayError(Exception):
    """An overlay that cannot safely be written into a slot."""


def resident_syms_script(resident_elf, entry, path):
    """A linker script assigning every resident symbol an overlay may bind to.

    The slot symbol itself is absolute -- it comes from the fragment the resident
    was built with -- and re-exporting it would collide with the overlay's own
    definition of the same name.
    """
    syms = defined_symbols(resident_elf)
    with open(path, "w") as f:
        f.write("/* Resident symbols an overlay may bind to. Generated. */\n")
        for name, addr in sorted(syms.items()):
            if name != entry:
                f.write(f"{name} = 0x{addr:x};\n")
    return path


def link(
    objects,
    resident,
    slot_base,
    slot_size,
    output,
    entry="overlay_entry",
    target="aie2p-none-unknown-elf",
    col=0,
    row=2,
    geometry=None,
):
    """Link objects into the slot and verify the result is usable."""
    resident_elf = find_core_elf(resident, col, row)
    syms_ld = resident_syms_script(resident_elf, entry, f"{output}.resident-syms.ld")

    # Place the entry first so it lands exactly on the address the resident
    # jumps to, whatever else the kernel drags in.
    script = f"{output}.overlay.ld"
    with open(script, "w") as f:
        f.write(
            f"SECTIONS {{\n"
            f"  .text 0x{slot_base:x} : {{\n"
            f"    *(.text.{entry})\n"
            f"    *(.text*)\n"
            f"  }}\n"
            f"}}\n"
            f"INPUT({syms_ld})\n"
        )

    cmd = [
        peano("clang"),
        "-O2",
        f"--target={target}",
        "-fuse-ld=lld",
        # No crt0 and no libc: those would collide with the resident's copies,
        # and an overlay is a function, not a program.
        "-nostdlib",
        "-nostartfiles",
        *objects,
        f"-Wl,-T,{script}",
        # Without an entry, --gc-sections has no root and silently emits an
        # empty .text.
        f"-Wl,--entry={entry}",
        "-Wl,--gc-sections",
        "-o",
        output,
    ]
    if subprocess.run(cmd).returncode:
        raise OverlayError(f"linking {' '.join(objects)} into the slot failed")

    # Record which resident symbols this overlay bound to, because the linked
    # overlay no longer says. Every import was resolved to an absolute address
    # here, so the finished ELF has no undefined symbols at all -- the
    # information exists only in the input objects, and only at this moment.
    # `check` needs it to notice a resident symbol that disappeared in pass 2,
    # which is invisible to a comparison of the symbols the two passes share.
    resident_syms = defined_symbols(resident_elf)
    imports = set()
    for obj in objects:
        imports |= undefined_symbols(obj) & resident_syms.keys()
    with open(f"{output}.imports", "w") as f:
        f.write("\n".join(sorted(imports)))

    return verify(output, slot_base, slot_size, entry, geometry)


def verify(path, slot_base, slot_size, entry="overlay_entry", geometry=None):
    """Reject an overlay that would load and then misbehave."""
    symbols, sections = read_elf(path)

    # Constructors first: they show up as an extra allocatable section, so the
    # generic check below would catch them with a message about .rodata that
    # says nothing about the actual problem.
    ctors = [(n, sz) for n, _, sz, fl, _ in sections if n in CTOR_SECTIONS and sz]
    if ctors:
        raise OverlayError(
            f"{path}: has static constructors ({', '.join(n for n, _ in ctors)}). "
            f"They would never run: crt0 walks the constructor list once when the "
            f"core starts, long before any overlay is written into the slot, so "
            f"the overlay would execute against uninitialized state."
        )

    alloc = [(n, a, s) for n, a, s, fl, _ in sections if fl & SHF_ALLOC and s]
    if not alloc:
        raise OverlayError(
            f"{path}: has no allocatable content at all. --gc-sections collected "
            f"everything, which usually means nothing is reachable from {entry} "
            f"and the padding lacks the `retain` attribute."
        )
    if len(alloc) != 1 or alloc[0][0] != ".text":
        raise OverlayError(
            f"{path}: expected exactly one allocatable section named .text, got "
            f"{[n for n, _, _ in alloc]}. An overlay is written into program "
            f"memory as a single block; .rodata, .data and .bss live in *data* "
            f"memory, which nothing swaps, so they would have to persist across "
            f"every overlay simultaneously."
        )

    _, addr, size = alloc[0]
    if addr != slot_base:
        raise OverlayError(
            f"{path}: .text is at 0x{addr:x}, not the slot 0x{slot_base:x}. The "
            f"resident jumps to a fixed address, so the core would jump into the "
            f"middle of something."
        )
    if size > slot_size:
        raise OverlayError(
            f"{path}: .text is {size} bytes, larger than the {slot_size}-byte "
            f"slot; writing it would run past the slot into whatever follows."
        )
    if size % PROG_MEM_LINE:
        raise OverlayError(
            f"{path}: .text is {size} bytes, not a multiple of a "
            f"{PROG_MEM_LINE}-byte program-memory line"
        )

    e = symbols.get(entry)
    if not e or e[0] != slot_base:
        raise OverlayError(
            f"{path}: {entry} is at "
            f"{'0x%x' % e[0] if e else 'nowhere'}, but the resident jumps to "
            f"0x{slot_base:x}"
        )

    # The geometry rules are the hardware ones, and they are checked against the
    # slot the overlay was actually linked into rather than assumed.
    if geometry is not None:
        try:
            geometry.validate()
        except GeometryError as exc:
            raise OverlayError(f"{path}: {exc}") from exc
        if not any(s.base == slot_base and size <= s.size for s in geometry.slots):
            raise OverlayError(
                f"{path}: .text at 0x{slot_base:x} ({size} bytes) does not fit any "
                f"slot in this geometry: "
                f"{', '.join(str(s) for s in geometry.slots)}"
            )

    return size
