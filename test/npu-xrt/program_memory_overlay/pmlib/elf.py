# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Just enough ELF32 to place and inspect overlays.

Peano ships no llvm-objcopy, so this reads the ELF with struct. It is also the
only reliable way to total section sizes: `llvm-readelf -S` prints the index as
`[ 3]` (two whitespace-separated fields) but `[10]` as one, so a column-based
awk silently reads the wrong field once a file has ten or more sections.
"""

import glob
import os
import struct
import sys

SHN_UNDEF = 0
SHN_ABS = 0xFFF1
SHF_ALLOC = 0x2
SHT_SYMTAB = 2

# Sections holding static constructors. This target emits .ctors; .init_array is
# the other spelling. Either means an overlay carries initialization that would
# never run, because crt0 walks the list once at core start -- long before any
# overlay reaches the slot.
CTOR_SECTIONS = (".ctors", ".init_array")


def read_elf(path):
    """Return (symbols, sections) of an ELF32-LE.

    symbols maps name -> (addr, size, shndx); sections is a list of
    (name, addr, size, flags, bytes). shndx distinguishes a defined symbol from
    an undefined (SHN_UNDEF) or absolute (SHN_ABS) one, which matters because
    address 0 is a perfectly ordinary place for resident code to live.
    """
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"\x7fELF" or blob[4] != 1 or blob[5] != 1:
        sys.exit(f"{path}: not a little-endian 32-bit ELF")

    e_shoff, _, _, _, _, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(
        "<IIHHHHHH", blob, 0x20
    )
    shdrs = [
        struct.unpack_from("<10I", blob, e_shoff + i * e_shentsize)
        for i in range(e_shnum)
    ]

    def name_at(strtab_off, off):
        end = blob.index(b"\0", strtab_off + off)
        return blob[strtab_off + off : end].decode()

    shstr = shdrs[e_shstrndx][4]
    sections, symbols = [], {}
    for sh in shdrs:
        nm, sh_type, flags, addr, off, size, link, _, _, entsize = sh
        body = b"" if sh_type == 8 else blob[off : off + size]  # 8 = SHT_NOBITS
        sections.append((name_at(shstr, nm), addr, size, flags, body))
        if sh_type != SHT_SYMTAB:
            continue
        stroff = shdrs[link][4]
        for i in range(size // entsize):
            st_name, st_value, st_size, _, _, st_shndx = struct.unpack_from(
                "<IIIBBH", blob, off + i * entsize
            )
            if st_name:
                symbols[name_at(stroff, st_name)] = (st_value, st_size, st_shndx)
    return symbols, sections


def section_file_offset(path, want):
    """Byte offset of a section's contents within the file."""
    with open(path, "rb") as f:
        blob = f.read()
    e_shoff, _, _, _, _, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(
        "<IIHHHHHH", blob, 0x20
    )
    shdrs = [
        struct.unpack_from("<10I", blob, e_shoff + i * e_shentsize)
        for i in range(e_shnum)
    ]
    shstr = shdrs[e_shstrndx][4]
    for nm, _, _, _, off, _, _, _, _, _ in shdrs:
        end = blob.index(b"\0", shstr + nm)
        if blob[shstr + nm : end].decode() == want:
            return off
    sys.exit(f"{path}: no section named {want}")


def text_size(path):
    """Total allocatable .text* bytes, the measure of program-memory footprint."""
    _, sections = read_elf(path)
    return sum(
        sz for n, _, sz, fl, _ in sections if n.startswith(".text") and fl & SHF_ALLOC
    )


def text_words(path):
    """The overlay's .text as 32-bit words, ready to write to program memory."""
    _, sections = read_elf(path)
    for name, _, size, flags, body in sections:
        if name == ".text" and flags & SHF_ALLOC and size:
            if len(body) < size:
                sys.exit(
                    f"{path}: .text is {size} bytes but holds no contents; an "
                    f"overlay cannot be written from a NOBITS section"
                )
            return list(struct.unpack(f"<{size // 4}I", body[:size]))
    sys.exit(f"{path}: no .text")


def stack_frames(path):
    """Per-function stack frame sizes from .stack_sizes, or None if absent.

    Needs -fstack-size-section at compile time. The section is a sequence of
    (address, ULEB128 frame size) pairs, the address being target-pointer sized
    -- four bytes here.

    This is the only way to see an overlay's stack demand from outside it. The
    resident's stack is sized when the resident links, and an overlay linked
    separately afterwards is invisible to that: nothing connects the two, so an
    overlay with a frame larger than the budget overruns into whatever sits
    below the stack. Silently -- the symptom is scattered wrong values in
    another buffer, not a fault.
    """
    _, sections = read_elf(path)
    blob = next((b for n, _, _, _, b in sections if n == ".stack_sizes"), None)
    if blob is None:
        return None

    frames, i = [], 0
    while i + 4 < len(blob):
        addr = int.from_bytes(blob[i : i + 4], "little")
        i += 4
        size, shift = 0, 0
        while i < len(blob):
            byte = blob[i]
            i += 1
            size |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        frames.append((addr, size))
    return frames


def max_stack_frame(path):
    """The largest single frame, or None if the object carries no sizes."""
    frames = stack_frames(path)
    return max((sz for _, sz in frames), default=0) if frames is not None else None


def defined_symbols(path):
    """Symbols an overlay may bind to: defined, neither undefined nor absolute."""
    symbols, _ = read_elf(path)
    return {
        n: addr
        for n, (addr, _, shndx) in symbols.items()
        if shndx not in (SHN_UNDEF, SHN_ABS)
    }


def undefined_symbols(path):
    """Symbols this object expects someone else to define."""
    symbols, _ = read_elf(path)
    return {n for n, (_, _, shndx) in symbols.items() if shndx == SHN_UNDEF}


def find_core_elf(path, col=0, row=2):
    """Resolve a core ELF, accepting an aiecc tmpdir as a shorthand.

    aiecc names the artifact after the device symbol, which the caller does not
    control, so search rather than hardcode. col/row must be threaded through by
    callers -- defaulting them silently is how a design on another tile ends up
    validated against the wrong core.
    """
    if os.path.isfile(path):
        return os.path.abspath(path)
    matches = sorted(
        glob.glob(os.path.join(path, "**", f"*core_{col}_{row}*.elf"), recursive=True)
    )
    if not matches:
        sys.exit(f"{path}: no core ({col},{row}) ELF found")
    if len(matches) > 1:
        sys.exit(f"{path}: ambiguous core ELFs for ({col},{row}): {matches}")
    return os.path.abspath(matches[0])


def peano(tool):
    root = os.environ.get("PEANO_INSTALL_DIR")
    if not root:
        sys.exit("PEANO_INSTALL_DIR is not set")
    return os.path.join(root, "bin", tool)
