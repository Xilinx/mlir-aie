//===- good-tct-nontoken-await-ignored.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops --canonicalize \
// RUN:         --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// The token-balance check only reasons about issue_token tasks. Awaiting a task
// that does NOT issue a token is a separate error owned by aie-dma-tasks-to-npu
// ("cannot wait on a BD not configured to issue a token"); the token-balance
// pass must not double-diagnose it, and its non-token starts/awaits must not
// perturb any channel's count. This constant-trip loop unrolls to straight-line
// non-token configure/start/await triples and allocates without a token error.

// CHECK-LABEL: @nontoken_await_ignored
// CHECK-NOT:   scf.for
// CHECK:       aie.dma_bd
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @nontoken_await_ignored(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
