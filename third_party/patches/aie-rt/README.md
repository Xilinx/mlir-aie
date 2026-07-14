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
