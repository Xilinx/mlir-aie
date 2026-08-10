<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Vendored aie-rt patches

`third_party/aie-rt` is pinned to a commit on upstream
[Xilinx/aie-rt](https://github.com/Xilinx/aie-rt) `release/main_aig`. The
patches in this directory carry functionality mlir-aie depends on that isn't
upstream yet, applied automatically at CMake configure time (see
`runtime_lib/xaiengine/aiert.cmake`).

- `0001-cdo-sim-defork-fixes.patch`: works around aie-rt's
  `cdo_rts.h`/`main_rts.h` dependencies on Vitis-only headers by replacing the
  includes with local forward declarations (`xaie_cdo.c`, `xaie_sim.c`), fixes
  a resource-manager memory leak (`RscArrPerTile` in `xaie_io_common.c`), and
  carries a few minor build/warning fixes. None of this is present upstream as
  of the pinned commit.
- `0002-elfloader-zero-bss-gap.patch`: `_XAie_LoadDataMemSection` wrote
  `p_memsz` bytes from a buffer (`ElfMem + p_offset`) holding only `p_filesz`
  valid ones, substituting a zeroed buffer only when `p_filesz == 0`. That
  covers a pure-data segment and a pure-bss segment, but not the ordinary mixed
  `.data`+`.bss` segment (`0 < p_filesz < p_memsz`) that any link with both
  initialised and zero-initialised globals produces -- for those it read past
  the segment's file contents and wrote the following ELF bytes (`.comment`,
  `.symtab`) into tile data memory, where zero-initialised statics live. The
  System V gABI defines those trailing bytes to hold zero; that is how `.bss` is
  represented in a loadable segment. The patch allocates a `p_memsz` zeroed
  buffer whenever `p_memsz > p_filesz` and copies the initialised prefix into
  it. Regression test: `test/aiecc/bss_zero_init.mlir`. Not fixed upstream as of
  the pinned commit.

