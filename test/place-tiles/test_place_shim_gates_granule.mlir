//===- test_place_shim_gates_granule.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s --check-prefix=SEQ
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SA

// Three ends receive on the one shim's two S2MM channels. @out1 has no proof,
// so @out and @out2 must take turns. @out's only proof that fits its drain is
// @inb's, through the join's later segment, which holds in whole 256-byte
// objects; @out drains 384 bytes, so allocation does not credit it and will
// not share. The placers read the same granule and see the shortfall.
// SEQ: error: no ShimNOCTile on the device has 1 input/0 output DMA channel(s) free
// SA:  warning: SA placer's resource estimate is over capacity

module @granule_unproven_turns {
  aie.device(npu2_1col) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %c2 = aie.logical_tile<CoreTile>(0, 4)
    %c3 = aie.logical_tile<CoreTile>(0, 5)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @ina(%s0, {%c0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @inb(%s0, {%c1}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%c0, {%mem}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @b(%c1, {%mem}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out(%mem, {%s0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 32] [])
    aie.objectfifo @out1(%c2, {%s1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2(%s2, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2(%c3, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c0) {
      %i = aie.objectfifo.acquire @ina(Consume, 1) : memref<32xi32>
      %o = aie.objectfifo.acquire @a(Produce, 1) : memref<32xi32>
      aie.objectfifo.release @ina(Consume, 1)
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    aie.core(%c1) {
      %i = aie.objectfifo.acquire @inb(Consume, 1) : memref<32xi32>
      %o = aie.objectfifo.acquire @b(Produce, 1) : memref<32xi32>
      aie.objectfifo.release @inb(Consume, 1)
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    aie.core(%c2) {
      %o = aie.objectfifo.acquire @out1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out1(Produce, 1)
      aie.end
    }
    aie.core(%c3) {
      %i = aie.objectfifo.acquire @in2(Consume, 1) : memref<16xi32>
      %o = aie.objectfifo.acquire @out2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in2(Consume, 1)
      aie.objectfifo.release @out2(Produce, 1)
      aie.end
    }
    aie.runtime_sequence @seq0(%x: memref<64xi32>, %y: memref<32xi32>, %z: memref<96xi32>) {
      %ta = aiex.dma_configure_task_for @ina {
        aie.dma_bd(%x : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      %tb = aiex.dma_configure_task_for @inb {
        aie.dma_bd(%y : memref<32xi32> offset = 0 len = 32)
        aie.end
      } {issue_token = true}
      %to = aiex.dma_configure_task_for @out {
        aie.dma_bd(%z : memref<96xi32> offset = 0 len = 96)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_start_task(%ta)
      aiex.dma_start_task(%tb)
      aiex.dma_await_task(%to)
      aiex.dma_await_task(%ta)
      aiex.dma_await_task(%tb)
    }
    aie.runtime_sequence @seq1(%a: memref<16xi32>, %b: memref<16xi32>) {
      %to = aiex.dma_configure_task_for @out1 {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
    }
    aie.runtime_sequence @seq2(%a: memref<16xi32>, %b: memref<16xi32>) {
      %ti = aiex.dma_configure_task_for @in2 {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      %to = aiex.dma_configure_task_for @out2 {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_start_task(%ti)
      aiex.dma_await_task(%to)
      aiex.dma_await_task(%ti)
    }
  }
}
