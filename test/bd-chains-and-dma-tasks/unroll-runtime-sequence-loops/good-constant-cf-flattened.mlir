//===- good-constant-cf-flattened.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops --canonicalize \
// RUN:         --split-input-file %s | FileCheck %s

// The static path invariant: in a runtime sequence with only compile-time
// control flow, no scf op reaches BD-ID allocation. Constant-trip scf.for is
// unrolled by --aie-unroll-runtime-sequence-loops; constant-predicate scf.if is
// folded by --canonicalize. These two passes run before
// --aie-assign-runtime-sequence-bd-ids, so the allocator only ever sees
// straight-line IR (runtime-valued control flow is instead rejected by the
// allocator's validate() gate for the dynamic path).



// -----

// Constant-predicate scf.if is folded away.
// CHECK-LABEL: @const_if
// CHECK-NOT:   scf.if
// CHECK:       aie.dma_bd
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @a(%tile_0_0, MM2S, 0)
  aie.runtime_sequence @const_if(%arg0: memref<8xi16>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    scf.if %true {
      %k = aiex.dma_configure_task_for @a {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%k)
      aiex.dma_await_task(%k)
    }
  }
}



// -----

// Constant scf.for nested in a constant scf.if: the for unrolls and the if
// folds, leaving pure straight-line IR (no scf op of either kind survives).
// CHECK-LABEL: @const_if_for
// CHECK-NOT:   scf.if
// CHECK-NOT:   scf.for
// CHECK:       aie.dma_bd
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @b(%tile_0_0, MM2S, 0)
  aie.runtime_sequence @const_if_for(%arg0: memref<8xi16>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.if %true {
      scf.for %i = %c0 to %c3 step %c1 {
        %k = aiex.dma_configure_task_for @b {
          aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%k)
        aiex.dma_await_task(%k)
      }
    }
  }
}
