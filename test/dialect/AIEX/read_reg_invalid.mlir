//===- read_reg_invalid.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -verify-diagnostics %s

// A non-tile SSA operand is rejected. (No out-of-range col/row test: the tile
// is an SSA aie.tile value, whose own verifier bounds it against the device,
// same as aiex.core_reset / aiex.dma_channel_reset.)
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      %c0 = arith.constant 0 : index
      // expected-error @+1 {{tile operand must be produced by an aie.tile op}}
      aiex.npu.read_reg(%c0, 0)
    }
  }
}
