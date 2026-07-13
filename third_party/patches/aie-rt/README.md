# Vendored aie-rt patches

`third_party/aie-rt` is pinned to a commit on upstream
[Xilinx/aie-rt](https://github.com/Xilinx/aie-rt) `release/main_aig`. The
patches in this directory carry functionality mlir-aie depends on that isn't
upstream yet, applied automatically at CMake configure time (see
`runtime_lib/xaiengine/aiert.cmake`).

- `0001-vck5000-cdo-sim-fixes.patch`: adds the VCK5000/AMDAIR IO backend
  (`xaie_amdair.c`, used by `runtime_lib/test_lib/test_library.cpp` for
  VCK5000 hardware testing), works around aie-rt's `cdo_rts.h`/`main_rts.h`
  header dependencies on Vitis-only headers, and fixes a resource-manager
  memory leak (`RscArrPerTile`). None of this is present upstream as of the
  pinned commit.
