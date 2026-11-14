//===- test_sa_ctrl_pkt_overlay.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// With ctrl-pkt-overlay, the SA placer counts the MM2S channel the control
// overlay takes on each shim tile, shared with the ends never in flight with
// its control packets, as objectFIFO allocation will.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1 ctrl-pkt-overlay=true' %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=2 ctrl-pkt-overlay=true' %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=3 ctrl-pkt-overlay=true' %s | FileCheck %s

// One column: both inputs and the overlay, kept apart by a control packet
// between the transfers, need three channels of the shim tile's two, which
// allocation then refuses too.
// CHECK-LABEL: module @overlay_one_column
module @overlay_one_column {
  // expected-warning@+1 {{SA placer's resource estimate is over capacity (penalty=20)}}
  aie.device(npu2_1col) {
    // expected-note@+1 {{DMA channels penalty=20 on (0, 0)}}
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}
