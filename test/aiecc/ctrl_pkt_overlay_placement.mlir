//===- ctrl_pkt_overlay_placement.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc tells placement about --generate-ctrl-pkt-overlay, so the placer
// leaves column 0's shim tile the overlay's channel, which a control packet
// between the inputs' transfers keeps from sharing with either of them.

// RUN: rm -rf %t && mkdir -p %t && cd %t && %aiecc -n --no-progress --tmpdir %t/prj --generate-ctrl-pkt-overlay --get=input_physical.mlir %s
// RUN: FileCheck %s --input-file=%t/input_physical.mlir

// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay, aie.shareable_with = []}
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @b__shim_alloc(%shim_noc_tile_1_0, MM2S, 1)

module @overlay_interleaved {
  aie.device(npu2) {
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
