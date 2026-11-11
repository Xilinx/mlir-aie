//===- ctrl_pkt_overlay_two_devices.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two configurations of one target share one control overlay, which
// aie-generate-column-control-overlay seats by every device's ends. @cfgA
// pins a circuit end on shim (0, 0) MM2S 0, so the overlay takes MM2S 1 of
// column 0, where @cfgB's packet end @p, pinned to MM2S 1 and in flight with
// a control packet, cannot share it. Both placers put @cfgB's logical shim
// on column 1.

// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=1})' %s | FileCheck %s

// CHECK-LABEL: aie.device(npu2_2col) @cfgB
// CHECK: aie.shim_dma_allocation @p__shim_alloc(%{{.*}}tile_1_0, MM2S, 1,

module @overlay_two_devices {
  aie.device(npu2_2col) @cfgA {
    %s = aie.tile(0, 0)
    %c = aie.tile(0, 2)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @runA(%x : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
    }
  }
  aie.device(npu2_2col) @cfgB {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.tile(0, 2)
    aie.objectfifo @p(%s, {%c}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @p(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @p(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @runB(%x : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @p, id = 0 : i64} : memref<16xi32>
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @p, id = 0 : i64} : memref<16xi32>
    }
  }
}
