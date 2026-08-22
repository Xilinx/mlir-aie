# overlay_elf.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# aie2.py's RUN lines drive this; lit must not pick it up as a test of its own, so
# disable it with a bogus requires line
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

"""Read the overlay pair out of a compiled core ELF.

Two entry points, both used by aie2.py:

  * ``overlay_pair`` (imported) turns a first-pass ELF into the patch aie2.py
    emits -- the victim's address, and the donor's bytes to write over it.
  * ``--check A B`` (command line) compares two builds and fails if the core
    moved between them, which would leave the patch aimed at the wrong address.

Kept free of MLIR so the ELF handling can be exercised on its own. Peano ships no
llvm-objcopy, so the ELF is parsed here with struct.
"""

import argparse
import glob
import os
import struct
import sys

# AIE2/AIE2P tile AXI-MM apertures, from
# third_party/aie-rt/driver/src/global/xaie2pgbl_params.h.
PROG_MEM_BASE = 0x20000
PROG_MEM_SIZE = 0x4000
# PROGRAM_MEMORY_ERROR_INJECTION: the same 16 KB with ECC checking disabled.
# Writing here instead of PROG_MEM_BASE tells an ECC failure apart from a plain
# write failure.
PROG_MEM_ECC_BYPASS_BASE = 0x24000
# PROGRAM_MEMORY_WIDTH is 128 bits, so this is both the ECC granule and the most
# a single control packet can carry.
PROG_MEM_LINE = 16

CORE_CONTROL = 0x32000  # bit 0 ENABLE, bit 1 RESET
DEBUG_CONTROL0 = 0x32010  # bit 0 DEBUG_HALT


def find_core_elf(path, col, row):
    """Resolve a core ELF, accepting an aiecc tmpdir as a shorthand.

    aiecc names the artifact after the device symbol, which the test does not
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


def _read_elf(path):
    """Return {name: (addr, size)} and the .text (addr, bytes) of an ELF32-LE."""
    with open(path, "rb") as f:
        blob = f.read()

    if blob[:4] != b"\x7fELF" or blob[4] != 1 or blob[5] != 1:
        sys.exit(f"{path}: not a little-endian 32-bit ELF")

    # e_shoff, e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum,
    # e_shstrndx -- the ELF32 header tail, starting at e_shoff.
    e_shoff, _, _, _, _, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(
        "<IIHHHHHH", blob, 0x20
    )
    shdrs = [
        struct.unpack_from("<10I", blob, e_shoff + i * e_shentsize)
        for i in range(e_shnum)
    ]

    def sh_name(sh):
        strtab_off = shdrs[e_shstrndx][4]
        end = blob.index(b"\0", strtab_off + sh[0])
        return blob[strtab_off + sh[0] : end].decode()

    symbols = {}
    text = None
    for sh in shdrs:
        _, sh_type, _, addr, off, size, link, _, _, entsize = sh
        if sh_name(sh) == ".text":
            text = (addr, blob[off : off + size])
        if sh_type != 2:  # SHT_SYMTAB
            continue
        str_off = shdrs[link][4]
        for i in range(size // entsize):
            st_name, st_value, st_size, _, _, _ = struct.unpack_from(
                "<IIIBBH", blob, off + i * entsize
            )
            if not st_name:
                continue
            end = blob.index(b"\0", str_off + st_name)
            symbols[blob[str_off + st_name : end].decode()] = (st_value, st_size)

    if text is None:
        sys.exit(f"{path}: no .text section")
    return symbols, text


def overlay_pair(elf_path, victim, donor):
    """Validate the overlay pair and return (victim address, donor words).

    The donor's bytes are copied verbatim over the victim, so the two have to be
    interchangeable: equal length, and the victim on a program-memory line
    boundary so the write covers whole lines and cannot force a read-modify-write
    of the ECC checkbits. Both must also be branch-free -- AIE2P encodes control
    transfers as absolute addresses, so a branch would need a relink -- which the
    caller gets by construction from how ovl.cc is written.
    """
    symbols, (text_addr, text_bytes) = _read_elf(elf_path)

    for name in (victim, donor):
        if name not in symbols:
            sys.exit(
                f"{elf_path}: {name} not found -- it was probably dropped by "
                f"--gc-sections. Check that ovl.cc still marks it "
                f"__attribute__((used, retain))."
            )

    v_addr, v_size = symbols[victim]
    d_addr, d_size = symbols[donor]

    if v_size != d_size:
        sys.exit(
            f"{victim} is {v_size} bytes but {donor} is {d_size}; the two must be "
            f"identical in size to be swapped without relinking"
        )
    if v_size % PROG_MEM_LINE or v_addr % PROG_MEM_LINE:
        sys.exit(
            f"{victim} is at 0x{v_addr:x} size {v_size}; both must be multiples of "
            f"{PROG_MEM_LINE} so the write covers whole program-memory lines"
        )
    if v_addr + v_size > PROG_MEM_SIZE:
        sys.exit(f"{victim} at 0x{v_addr:x} is past the end of program memory")

    start = d_addr - text_addr
    body = text_bytes[start : start + d_size]
    if len(body) != d_size:
        sys.exit(f"{donor} at 0x{d_addr:x} is not contained in .text")

    return v_addr, list(struct.unpack(f"<{d_size // 4}I", body))


def main():
    """`--check A B`: the two builds agree on where the patch goes and what it is.

    The design is emitted twice -- once to produce the core ELF, once with the
    patch derived from it -- and the second build recompiles the core rather than
    reusing the first ELF. That is fine only as long as the core lands at the
    same address both times, which this asserts instead of assuming.
    """
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--check", nargs=2, required=True, metavar=("BUILD_A", "BUILD_B"))
    p.add_argument("--col", type=int, default=0)
    p.add_argument("--row", type=int, default=2)
    args = p.parse_args()

    elfs = [find_core_elf(d, args.col, args.row) for d in args.check]

    for pair in ("near", "far"):
        victim, donor = f"sel_{pair}_a", f"sel_{pair}_b"
        a, b = (overlay_pair(e, victim, donor) for e in elfs)
        if a != b:
            sys.exit(
                f"core drifted between builds: {args.check[0]} put {victim} at "
                f"0x{a[0]:x} with patch {[hex(w) for w in a[1]]}, but "
                f"{args.check[1]} put it at 0x{b[0]:x} with "
                f"{[hex(w) for w in b[1]]}. The emitted patch targets the wrong "
                f"address."
            )

    # The whole point of the near/far split is the distance from the spin loop,
    # so report it rather than trusting the source ordering to have worked.
    symbols, _ = _read_elf(elfs[0])
    spin = symbols["ovl_wait"][0]
    dists = {pair: abs(symbols[f"sel_{pair}_a"][0] - spin) for pair in ("near", "far")}
    if dists["far"] <= dists["near"]:
        sys.exit(
            f"far pair is not farther from ovl_wait than the near pair "
            f"({dists}); the filler in ovl.cc is not separating them"
        )
    print(
        f"core stable: sel_near_a {dists['near']} bytes from ovl_wait, "
        f"sel_far_a {dists['far']} bytes"
    )


if __name__ == "__main__":
    main()
