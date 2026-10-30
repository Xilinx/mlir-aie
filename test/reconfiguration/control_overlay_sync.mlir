// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-create-control-overlay="adaptive=false" %s | FileCheck %s

// CHECK-LABEL: @synced
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CHECK-SAME: aie.shareable_with = [@a_prod_dma, @b_prod_dma]
// CHECK-LABEL: @unsynced
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CHECK-SAME: aie.shareable_with = []

module {
  aie.device(npu2_1col) @synced {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim, MM2S, 0)
    aie.objectfifo @a(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @ctrlpkt_col0_mm2s_chan0, id = 2 : i64, issue_token = true} : memref<16xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}


// -----

module {
  aie.device(npu2_1col) @unsynced {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim, MM2S, 0)
    aie.objectfifo @a(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @ctrlpkt_col0_mm2s_chan0, id = 2 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}

// -----

// CHECK-LABEL: @partially_synced
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CHECK-SAME: aie.shareable_with = []

module {
  aie.device(npu2_1col) @partially_synced {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim, MM2S, 0)
    aie.objectfifo @a(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @ctrlpkt_col0_mm2s_chan0, id = 2 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @ctrlpkt_col0_mm2s_chan0, id = 3 : i64, issue_token = true} : memref<16xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}
