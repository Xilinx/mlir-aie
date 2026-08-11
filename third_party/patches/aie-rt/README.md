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
  includes with local forward declarations (`xaie_cdo.c`, `xaie_sim.c`), adds
  an explicit backend-selection parameter to `XAie_IOInit` and a matching
  `Backend` field on `XAie_Config` (`xaiegbl.c`/`.h`, `xaie_io.c`/`.h`) that
  `lib/Targets/AIERT.cpp` depends on to select the CDO backend, and carries a
  few minor build/warning fixes. None of this is present upstream as of the
  pinned commit.
- `0002-remove-dead-blockwrite32-append-fns.patch`: deletes
  `_XAie_AppendBlockWrite32`/`_opt` in `xaie_txn.c`, two `static inline`
  functions left over from an upstream TXN-serialization refactor that no
  caller references anymore (block-write commands now go through
  `_XAie_AppendBWToTxnBuff`/`_XAie_AppendBWToBlockwriteBuff` instead). GCC
  doesn't flag this, but clang's `-Werror=unused` (used in CI) does.
- `0003-fix-aie1-tiledma-intrleavecount-underflow.patch`: fixes an unsigned
  underflow in the AIE1 tile-DMA BD writer. `_XAie_TileDmaWriteBd` guards each
  BD field with `_XAie_CheckPrecisionExceeds(Lsb, _XAie_MaxBitsNeeded(v), 32)`,
  and passes `IntrleaveCount - 1U` for the interleave-count field. Interleaving
  is off by default, so `IntrleaveCount` is 0 and the subtraction wraps to
  `0xFFFFFFFF`, making `_XAie_MaxBitsNeeded` return 32 and the guard trip for
  any nonzero `Lsb`. The BD write then bails out with `XAIE_ERR` before
  programming a single word, so every AIE1 tile DMA silently goes
  unconfigured. The neighbouring `XAie_SetField` that actually writes the
  field masks the wrapped value, which is why only the new check is affected.
  Clamp the checked value to 0 when interleaving is disabled. Reported
  upstream; drop this once it lands.
