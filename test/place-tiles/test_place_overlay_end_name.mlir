//===- test_place_overlay_end_name.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles='ctrl-pkt-overlay=true' %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1 ctrl-pkt-overlay=true' %s | FileCheck %s

// test_sa_ctrl_pkt_overlay.mlir's @overlay_full_shim: the control overlay
// shares a shim MM2S channel with an input never in flight with its control
// packets. The placers count the overlay's channel as an end no fifo is,
// whatever the fifo is named, so one named @ctrlpkt_col0 places alike.
// CHECK-LABEL: module @named_in0
// CHECK: aie.tile(0, 0)
// CHECK-LABEL: module @named_like_overlay
// CHECK: aie.tile(0, 0)

module @named_in0 {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @in0(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @in1(Consume, 1) : memref<16xi32>
      %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Consume, 1)
      aie.objectfifo.release @in1(Consume, 1)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%x : memref<16xi32>, %y : memref<16xi32>, %z : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in0, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in1, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%z[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out, id = 2 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

// -----

module @named_like_overlay {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @ctrlpkt_col0(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @ctrlpkt_col0(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @in1(Consume, 1) : memref<16xi32>
      %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @ctrlpkt_col0(Consume, 1)
      aie.objectfifo.release @in1(Consume, 1)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%x : memref<16xi32>, %y : memref<16xi32>, %z : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @ctrlpkt_col0, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in1, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%z[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out, id = 2 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
