#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# aie2.py's RUN lines drive this; lit must not pick it up as a test of its own, so
# disable it with a bogus requires line
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

"""Link an AIE kernel into a fixed program-memory slot, and read it back out.

An overlay is an ordinary object linked at the slot's address instead of at 0,
against the resident image's symbols so it can call back into code that is
always present. The result is verified to be exactly one allocatable section, at
the address the resident jumps to and no larger than the slot, because every one
of those going wrong produces a program that loads and then behaves oddly rather
than a build failure.

Peano ships no llvm-objcopy, so the ELF is read here with struct.
"""

import argparse
import glob
import os
import struct
import subprocess
import sys

PROG_MEM_LINE = 16  # program memory is 128 bits wide
SHN_UNDEF = 0
SHN_ABS = 0xFFF1


def _read_elf(path):
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
        sections.append((name_at(shstr, nm), addr, size, flags, blob[off : off + size]))
        if sh_type != 2:  # SHT_SYMTAB
            continue
        stroff = shdrs[link][4]
        for i in range(size // entsize):
            st_name, st_value, st_size, st_info, _, st_shndx = struct.unpack_from(
                "<IIIBBH", blob, off + i * entsize
            )
            if st_name:
                symbols[name_at(stroff, st_name)] = (st_value, st_size, st_shndx)
    return symbols, sections


def find_core_elf(path, col=0, row=2):
    """Resolve a core ELF, accepting an aiecc tmpdir as a shorthand.

    aiecc names the artifact after the device symbol, which the caller does not
    control, so search rather than hardcode.
    """
    if os.path.isfile(path):
        return os.path.abspath(path)
    matches = sorted(
        glob.glob(os.path.join(path, "**", f"*core_{col}_{row}*.elf"), recursive=True)
    )
    if not matches:
        sys.exit(f"{path}: no core ({col},{row}) ELF found")
    if len(matches) > 1:
        sys.exit(f"{path}: ambiguous core ELFs: {matches}")
    return os.path.abspath(matches[0])


def _peano(tool):
    root = os.environ.get("PEANO_INSTALL_DIR")
    if not root:
        sys.exit("PEANO_INSTALL_DIR is not set")
    return os.path.join(root, "bin", tool)


def link(args):
    """Link one kernel object into the slot and check the result is usable."""
    symbols, _ = _read_elf(find_core_elf(args.resident))

    # Feed the overlay only the resident's *defined* symbols. The slot symbol
    # itself is absolute -- it comes from the linker-script fragment the resident
    # was built with -- and re-exporting it here would collide with the overlay's
    # own definition of the same name.
    syms_ld = f"{args.output}.resident-syms.ld"
    with open(syms_ld, "w") as f:
        f.write("/* Resident symbols an overlay may call. Generated. */\n")
        for name, (addr, _, shndx) in sorted(symbols.items()):
            if shndx in (SHN_UNDEF, SHN_ABS) or name == args.entry:
                continue
            f.write(f"{name} = 0x{addr:x};\n")

    # Place the entry first so it lands exactly on the slot address the resident
    # jumps to, whatever else the kernel drags in.
    script = f"{args.output}.overlay.ld"
    with open(script, "w") as f:
        f.write(
            f"SECTIONS {{\n"
            f"  .text 0x{args.slot:x} : {{\n"
            f"    *(.text.{args.entry})\n"
            f"    *(.text*)\n"
            f"  }}\n"
            f"}}\n"
            f"INPUT({syms_ld})\n"
        )

    cmd = [
        _peano("clang"),
        "-O2",
        f"--target={args.target}",
        "-fuse-ld=lld",
        # No crt0 and no libc: those would collide with the resident's copies,
        # and an overlay is a function, not a program.
        "-nostdlib",
        "-nostartfiles",
        *args.object,
        f"-Wl,-T,{script}",
        # Without an entry, --gc-sections has no root and silently produces an
        # empty .text.
        f"-Wl,--entry={args.entry}",
        "-Wl,--gc-sections",
        "-o",
        args.output,
    ]
    if subprocess.run(cmd).returncode:
        sys.exit(f"linking {' '.join(args.object)} into the slot failed")

    _verify(args)


def _verify(args):
    symbols, sections = _read_elf(args.output)
    SHF_ALLOC = 0x2

    alloc = [(n, a, s) for n, a, s, fl, _ in sections if fl & SHF_ALLOC and s]
    if len(alloc) != 1 or alloc[0][0] != ".text":
        sys.exit(
            f"{args.output}: expected exactly one allocatable section named "
            f".text, got {[n for n, _, _ in alloc]}. An overlay is written into "
            f"program memory as a single block, so anything else -- .rodata, "
            f".data -- would need a home in data memory that persists across "
            f"every overlay."
        )

    _, addr, size = alloc[0]
    if addr != args.slot:
        sys.exit(f"{args.output}: .text is at 0x{addr:x}, not the slot 0x{args.slot:x}")
    if size > args.slot_size:
        sys.exit(
            f"{args.output}: .text is {size} bytes, larger than the "
            f"{args.slot_size}-byte slot"
        )
    if size % PROG_MEM_LINE:
        sys.exit(
            f"{args.output}: .text is {size} bytes, not a multiple of a "
            f"{PROG_MEM_LINE}-byte program memory line"
        )

    entry = symbols.get(args.entry)
    if not entry or entry[0] != args.slot:
        sys.exit(
            f"{args.output}: {args.entry} is at "
            f"{'0x%x' % entry[0] if entry else 'nowhere'}, but the resident jumps "
            f"to 0x{args.slot:x}"
        )
    if "_init_array_start" in symbols and symbols.get(
        "__init_array_end"
    ) != symbols.get("__init_array_start"):
        sys.exit(f"{args.output}: has static constructors, which would never run")

    print(f"{os.path.basename(args.output)}: {size} bytes at 0x{addr:x}")


def text_words(path):
    """The overlay's .text as 32-bit words, ready to be written to program memory."""
    _, sections = _read_elf(path)
    for name, _, size, flags, body in sections:
        if name == ".text" and flags & 0x2 and size:
            return list(struct.unpack(f"<{size // 4}I", body[:size]))
    sys.exit(f"{path}: no .text")


def check(args):
    """The resident the overlays were linked against is the one that got built.

    Both passes emit the same design, but pass 2 recompiles the core rather than
    reusing pass 1's ELF. Every overlay holds pass 1's addresses for the resident
    symbols it calls, so if the core moved between passes those calls now land on
    whatever happens to be there. Assert instead of assuming.
    """
    a, _ = _read_elf(find_core_elf(args.a))
    b, _ = _read_elf(find_core_elf(args.b))
    moved = [
        name
        for name, (addr, _, shndx) in a.items()
        if shndx not in (SHN_UNDEF, SHN_ABS)
        and name in b
        and b[name][0] != addr
        and not name.startswith(".L")
    ]
    if moved:
        sys.exit(
            f"the resident moved between passes: {sorted(moved)[:8]} changed "
            f"address, so every overlay calling them is now aimed at the wrong "
            f"code"
        )
    print(f"resident stable across passes ({len(a)} symbols)")


def sizes(args):
    """Report what the design would need if it were all resident at once.

    The point of overlays is that this total is not bounded by program memory.
    Printing it keeps the claim honest: if a design's total is comfortably under
    the limit, overlays are not buying it anything yet.
    """
    _, sections = _read_elf(find_core_elf(args.resident))
    resident = sum(sz for n, _, sz, fl, _ in sections if n == ".text" and fl & 0x2)

    total = resident
    print(f"  resident            {resident:6d} bytes")
    for e in args.overlays:
        n = len(text_words(e)) * 4
        total += n
        print(f"  {os.path.basename(e):<18} {n:6d} bytes")
    print(
        f"  {'total':<18} {total:6d} bytes of {args.program_memory} "
        f"({100.0 * total / args.program_memory:.0f}% of program memory)"
    )

    if args.require_exceeds and total <= args.program_memory:
        sys.exit(
            f"this design totals {total} bytes, which still fits in "
            f"{args.program_memory}: it no longer demonstrates a program larger "
            f"than program memory. Add or enlarge a kernel."
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    lk = sub.add_parser("link", help="link a kernel object into the slot")
    lk.add_argument(
        "--object",
        required=True,
        action="append",
        help="object to link into the slot; repeat for the wrapper plus the "
        "kernel it calls",
    )
    lk.add_argument(
        "--resident",
        required=True,
        help="the resident core ELF, or the aiecc tmpdir holding it",
    )
    lk.add_argument("--slot", required=True, type=lambda v: int(v, 0))
    lk.add_argument("--slot-size", required=True, type=lambda v: int(v, 0))
    lk.add_argument("--entry", default="overlay_entry")
    lk.add_argument("--target", default="aie2p-none-unknown-elf")
    lk.add_argument("--output", required=True)
    lk.set_defaults(func=link)

    sz = sub.add_parser("sizes", help="report resident + overlay code size")
    sz.add_argument("--resident", required=True)
    sz.add_argument("--overlays", required=True, nargs="+")
    sz.add_argument("--program-memory", type=lambda v: int(v, 0), default=0x4000)
    sz.add_argument(
        "--require-exceeds",
        action="store_true",
        help="fail unless the total exceeds program memory",
    )
    sz.set_defaults(func=sizes)

    ck = sub.add_parser("check", help="the resident did not move between passes")
    ck.add_argument("a", help="pass 1 core ELF or aiecc tmpdir")
    ck.add_argument("b", help="pass 2 core ELF or aiecc tmpdir")
    ck.set_defaults(func=check)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
