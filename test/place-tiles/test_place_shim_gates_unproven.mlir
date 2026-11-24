//===- test_place_shim_gates_unproven.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s --check-prefix=SEQ
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SA

// Three cores send to the host through one shim tile's two S2MM channels.
// The runtime sequence awaits each transfer before starting the next, but
// nothing proves the cores send only in their turn (no #aie.consumed_by),
// so objectFIFO allocation will not let two receiving ends share a channel.
// The placers count turns with the gates allocation uses, taken from
// aie-objectfifo-verify on a copy of the design, and see the shortfall.
// SEQ: error: no ShimNOCTile on the device has 1 input/0 output DMA channel(s) free
// SA:  warning: SA placer's resource estimate is over capacity

module @unproven_turns {
  aie.device(npu2_1col) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %c2 = aie.logical_tile<CoreTile>(0, 4)
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @out0(%c0, {%s0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%c1, {%s1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2(%c2, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c0) {
      %0 = aie.objectfifo.acquire @out0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out0(Produce, 1)
      aie.end
    }
    aie.core(%c1) {
      %0 = aie.objectfifo.acquire @out1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out1(Produce, 1)
      aie.end
    }
    aie.core(%c2) {
      %0 = aie.objectfifo.acquire @out2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out2(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%a: memref<16xi32>, %b: memref<16xi32>, %c: memref<16xi32>) {
      %t0 = aiex.dma_configure_task_for @out0 {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      %t1 = aiex.dma_configure_task_for @out1 {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      %t2 = aiex.dma_configure_task_for @out2 {
        aie.dma_bd(%c : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
    }
  }
}
