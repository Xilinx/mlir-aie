# format.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

"""On-disk layout of the AIE hsaco section, shared by the packer and the dumper.

This is a deliberate mirror of ROCr's ``core/inc/amd_aie_section.h`` (in the
rocm-systems repo, under ``projects/rocr-runtime/runtime/hsa-runtime/``).

Because the two now live in separate repositories the mirror can drift
silently, so :func:`hsaco.dump.parse_section` refuses a section whose
``version_major`` it does not recognise rather than misreading it.
"""

import struct

# Section magic: 'A','I','E','K' little-endian. Must match kAieSectionMagic.
MAGIC = 0x4B454941
VERSION_MAJOR = 1
VERSION_MINOR = 0

ARCHES = ("aie2", "aie2p")

# aie_section_header: magic, version_major, version_minor, header_size, kernel_count,
# kernel_entry_size, string_table_offset, string_table_size, blob_pool_offset, reserved[4].
HDR = "<IHHIIIIII" + "IIII"
HDR_SIZE = struct.calcsize(HDR)

# aie_kernel_entry: name_offset, insts_offset, insts_size, pdi_offset, pdi_size, kernarg_size,
# num_cols, kind, reserved[3].
ENTRY = "<IIIIIII" + "IIII"
ENTRY_SIZE = struct.calcsize(ENTRY)

# AieKernelKind. KIND_COUNT is the validation bound: anything at or above it is rejected, which
# is what keeps a future kind from being mis-read by a tool that predates it. The enum's
# Undecided shares Count's value but is runtime-only and never appears on disk.
KIND_PDI_INSTS = 0
KIND_FULL_ELF = 1
KIND_COUNT = 2
KIND_NAMES = {KIND_PDI_INSTS: "PdiInsts", KIND_FULL_ELF: "FullElf"}
