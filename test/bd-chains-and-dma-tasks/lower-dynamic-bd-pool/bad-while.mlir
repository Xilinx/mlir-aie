//===- bad-while.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s

// scf.while has no bounded trip count, so a BD id popped in its body has no
// deterministic push site; the dynamic pool path supports only scf.for and
// scf.if.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @dyn_while(%arg0: memref<1024xi32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{scf.while in a runtime sequence is not supported by the dynamic BD pool path}}
    scf.while (%i = %c0) : (i32) -> (i32) {
      %cmp = arith.cmpi slt, %i, %n : i32
      scf.condition(%cmp) %i : i32
    } do {
    ^bb0(%i: i32):
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
      %c1 = arith.constant 1 : i32
      %next = arith.addi %i, %c1 : i32
      scf.yield %next : i32
    }
  }
}
