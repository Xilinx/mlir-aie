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
# Program memory behaves as two halves; a write to the half the core is fetching
# from races. An overlay slot -- and so any block written as one -- must stay
# inside one half. See README.md.
PROG_MEM_HALF = 0x2000

# Distances in bytes from ovl_wait (where the core spins) to each overlay pair.
# ovl.cc lays the pairs out to hit these exactly; check_pairs() verifies it.
PAIR_DISTANCES = (
    64,
    384,
    512,
    640,
    768,
    896,
    960,
    1024,
    1152,
    1280,
    1408,
    2048,
    4160,
    8320,
)

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


def overlay_block(elf_path, victim, donor, block_bytes):
    """A whole-block patch containing the pair, for a realistically sized write.

    The 32-byte pair alone is the smallest observable write; a real overlay load
    moves kilobytes, takes proportionally longer, and has correspondingly more
    opportunity to collide with instruction fetch. This returns the program
    memory around the pair with the donor's bytes substituted in, so the write is
    block-sized while the observable stays exactly the same.

    Returns (block address, words), block-aligned and clipped to .text.
    """
    v_addr, donor_words = overlay_pair(elf_path, victim, donor)
    symbols, (text_addr, text_bytes) = _read_elf(elf_path)
    donor_bytes = len(donor_words) * 4

    if block_bytes < donor_bytes or block_bytes % PROG_MEM_LINE:
        sys.exit(
            f"block size {block_bytes} must be at least {donor_bytes} and a "
            f"multiple of {PROG_MEM_LINE}"
        )

    # Centre the block on the pair, then clamp it into both .text (so every byte
    # written is real code) and the victim's half of program memory (so the write
    # cannot spill into the half the core may be executing from).
    half_lo = (v_addr // PROG_MEM_HALF) * PROG_MEM_HALF
    lo = max(text_addr, half_lo)
    hi = min(text_addr + len(text_bytes), half_lo + PROG_MEM_HALF)
    if hi - lo < block_bytes:
        sys.exit(
            f"a {block_bytes}-byte block around {victim} does not fit in "
            f"[{lo}, {hi}) -- the half it lives in, clipped to .text"
        )
    start = v_addr - (block_bytes - donor_bytes) // 2
    start = max(lo, min(start, hi - block_bytes))
    start -= start % PROG_MEM_LINE

    body = bytearray(text_bytes[start - text_addr : start - text_addr + block_bytes])
    off = v_addr - start
    body[off : off + donor_bytes] = struct.pack(f"<{len(donor_words)}I", *donor_words)
    return start, list(struct.unpack(f"<{block_bytes // 4}I", bytes(body)))


def check_pairs(elfs):
    """Every pair is interchangeable in both builds and sits where it claims to."""
    for d in PAIR_DISTANCES:
        victim, donor = f"sel_d{d}_a", f"sel_d{d}_b"
        vals = [overlay_pair(e, victim, donor) for e in elfs]
        if len(set(map(str, vals))) != 1:
            sys.exit(
                f"core drifted between builds: {victim} is not identical across "
                f"{elfs}; the emitted patch would target the wrong address"
            )

    # The distances are the whole point, so verify rather than trust ovl.cc's
    # filler arithmetic to have survived a compiler change.
    symbols, _ = _read_elf(elfs[0])
    spin = symbols["ovl_wait"][0]
    for d in PAIR_DISTANCES:
        actual = spin - symbols[f"sel_d{d}_a"][0]
        if actual != d:
            sys.exit(
                f"sel_d{d}_a is {actual} bytes from ovl_wait, not {d}; the "
                f"filler in ovl.cc no longer produces the intended spacing"
            )
    print(f"core stable; pair distances verified: {list(PAIR_DISTANCES)}")


def main():
    """`--check A B`: the two builds agree on where each patch goes and what it is.

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

    check_pairs([find_core_elf(d, args.col, args.row) for d in args.check])


if __name__ == "__main__":
    main()
