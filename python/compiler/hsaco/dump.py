# dump.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

"""Parse and validate a hsaco section; print a human-readable summary."""

import argparse
import struct
import sys

from .elf import ElfFile
from .format import (
    ARCHES,
    ENTRY,
    ENTRY_SIZE,
    HDR,
    HDR_SIZE,
    KIND_COUNT,
    KIND_FULL_ELF,
    KIND_NAMES,
    MAGIC,
    VERSION_MAJOR,
)


def parse_section(section):
    """Return a validated description of a packed arch section.

    Every offset is bounds-checked against the section, so a truncated or
    corrupt section raises here rather than producing plausible-looking
    garbage.

    Args:
        section (bytes): The raw arch section.

    Returns:
        dict: ``arch_version``, ``kernel_count`` and a ``kernels`` list.

    Raises:
        ValueError: On bad magic, an unrecognised major version, or any field
            that does not lie within the section.
    """
    if len(section) < HDR_SIZE:
        raise ValueError("section smaller than header")
    magic, vmaj, vmin, hdr_size, kcount, kentry, st_off, st_size, pool_off, *_res = (
        struct.unpack_from(HDR, section, 0)
    )
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")
    # The layout is mirrored from a header in another repository (see
    # hsaco.format); refuse a major version this tool predates rather than
    # misreading its kernel table.
    if vmaj != VERSION_MAJOR:
        raise ValueError(
            f"unsupported section version {vmaj}.{vmin}; this tool understands "
            f"{VERSION_MAJOR}.x"
        )
    if kentry < ENTRY_SIZE:
        raise ValueError("kernel_entry_size too small")
    if hdr_size < HDR_SIZE:
        raise ValueError("header_size smaller than the header")
    if hdr_size + kcount * kentry > len(section):
        raise ValueError("kernel table out of bounds")
    if st_off + st_size > len(section):
        raise ValueError("string table out of bounds")
    if pool_off > len(section):
        raise ValueError("blob pool out of bounds")

    def in_pool(off, ln):
        # Every offset here comes from the section being validated, so none of
        # them is taken on trust: a blob bounded only from above could point
        # back at the metadata and still be reported as a valid kernel.
        # blob_pool_offset exists to give the lower bound.
        return off >= pool_off and off + ln <= len(section)

    kernels = []
    for i in range(kcount):
        base = hdr_size + i * kentry
        (
            name_off,
            insts_off,
            insts_size,
            pdi_off,
            pdi_size,
            kernarg_size,
            num_cols,
            kind,
            *_e,
        ) = struct.unpack_from(ENTRY, section, base)
        if kind >= KIND_COUNT:
            raise ValueError(f"kernel {i}: unknown kind {kind}")
        if kind == KIND_FULL_ELF and pdi_size:
            raise ValueError(f"kernel {i}: FullElf entry carries a separate PDI")
        if insts_size == 0 or not in_pool(insts_off, insts_size):
            raise ValueError(f"kernel {i}: insts out of bounds/overrun")
        if pdi_size and not in_pool(pdi_off, pdi_size):
            raise ValueError(f"kernel {i}: pdi out of bounds/overrun")
        # Bound the name against the string table, not merely against the
        # section: an offset past the table would otherwise read a name
        # straight out of the blob pool and report it as valid.
        if name_off >= st_size:
            raise ValueError(f"kernel {i}: name offset outside the string table")
        name_abs = st_off + name_off
        end = section.find(b"\x00", name_abs, st_off + st_size)
        if end == -1:
            raise ValueError(f"kernel {i}: name not terminated")
        kernels.append(
            {
                "name": section[name_abs:end].decode(),
                "insts_offset": insts_off,
                "insts_size": insts_size,
                "has_pdi": bool(pdi_size),
                "pdi_offset": pdi_off,
                "pdi_size": pdi_size,
                "kernarg_size": kernarg_size,
                "num_cols": num_cols,
                "kind": kind,
                "kind_name": KIND_NAMES.get(kind, "?"),
            }
        )
    return {"arch_version": (vmaj, vmin), "kernel_count": kcount, "kernels": kernels}


def read_sections_from_hsaco(path):
    """Return ``[(arch, section_bytes), ...]`` for every arch section in an hsaco.

    One hsaco can carry a section per architecture -- ROCr picks the one
    matching the running device -- so all of them are returned, in
    :data:`hsaco.format.ARCHES` order.

    Raises:
        ValueError: If the hsaco carries no aie2/aie2p section at all.
    """
    with open(path, "rb") as f:
        elf = ElfFile(f.read())
    sections = [
        (arch, section.data)
        for arch, section in ((a, elf.section_by_name(a)) for a in ARCHES)
        if section is not None
    ]
    if not sections:
        raise ValueError(f"{path}: no {'/'.join(ARCHES)} section found")
    return sections


def main(argv=None):
    """Print a summary of every arch section in an hsaco."""
    ap = argparse.ArgumentParser(
        prog="aie-hsaco-dump",
        description="Parse and validate the aie2/aie2p sections of an hsaco.",
    )
    ap.add_argument("--hsaco", required=True)
    args = ap.parse_args(argv)
    # A malformed hsaco is this tool's normal subject, not a crash: report the
    # carefully-worded ValueError rather than letting it out as a traceback.
    try:
        sections = read_sections_from_hsaco(args.hsaco)
    except (OSError, ValueError) as e:
        print(f"{ap.prog}: error: {e}", file=sys.stderr)
        return 1

    status = 0
    for arch, data in sections:
        print(f"arch section: {arch}")
        # Per section, so one damaged arch does not hide the intact ones --
        # which of the two is readable is exactly what the user needs to know.
        try:
            info = parse_section(data)
        except ValueError as e:
            print(f"{ap.prog}: error: {arch}: {e}", file=sys.stderr)
            status = 1
            continue
        print(f"version: {info['arch_version'][0]}.{info['arch_version'][1]}")
        for k in info["kernels"]:
            print(
                f"  kernel {k['name']}: kind={k['kind_name']} insts={k['insts_size']}B "
                f"pdi={'yes' if k['has_pdi'] else 'no'} "
                f"kernarg={k['kernarg_size']} cols={k['num_cols']}"
            )
    return status


if __name__ == "__main__":
    sys.exit(main())
