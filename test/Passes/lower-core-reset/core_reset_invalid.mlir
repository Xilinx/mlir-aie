//===- core_reset_invalid.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics --aie-lower-core-reset %s

// A mem tile is rejected: only core tiles have a CORE_CONTROL register with a
// reset bit. On npu2 row 1 is a mem tile, which has no compute core to reset.
// This matches aie-rt's XAie_CoreReset, which errors on non core tiles.
// (An out-of-range coordinate needs no test here: the tile is an SSA aie.tile
// value, whose own verifier bounds the column/row against the device.)
module {
  aie.device(npu2) {
    %mem_tile = aie.tile(0, 1)
    aie.runtime_sequence() {
      // expected-error @+1 {{tile (0, 1) has no core to reset (only core tiles have a CORE_CONTROL register)}}
      aiex.core_reset(%mem_tile)
    }
  }
}

// -----

// A shim NOC tile is rejected for the same reason: on npu2 row 0 is a shim tile,
// which has no compute core to reset.
module {
  aie.device(npu2) {
    %shim_tile = aie.tile(0, 0)
    aie.runtime_sequence() {
      // expected-error @+1 {{tile (0, 0) has no core to reset (only core tiles have a CORE_CONTROL register)}}
      aiex.core_reset(%shim_tile)
    }
  }
}

// -----

// AIE1 is rejected: the op lowers to an NPU control-packet write and the runtime
// sequence has no meaning on AIE1, matching SetLockOp's AIE1 rejection.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(0, 3)
    aie.runtime_sequence() {
      // expected-error @+1 {{not supported on AIE1}}
      aiex.core_reset(%tile)
    }
  }
}
