//===- ctrl_pkt_overlay_packet_flow.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aie-generate-column-control-overlay counts a shim packet flow's MM2S
// channel as packet demand it may share, not a circuit. With the overlay on
// channel 0 (adaptive off), it shares MM2S 0 with the packet flow and the
// logical shim's fifo takes MM2S 1; both placers place the shim there, as
// the overlay pass weighs the channel. Adaptive on gives the same.

// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true ctrlpkt-adaptive-overlay=false})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true ctrlpkt-adaptive-overlay=false placer=sa_placer sa-seed=1})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true})' %s | FileCheck %s --check-prefix=ADAPTIVE
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=1})' %s | FileCheck %s --check-prefix=ADAPTIVE

// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}, MM2S, 1)
// ADAPTIVE-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan
// ADAPTIVE-DAG: aie.shim_dma_allocation @a__shim_alloc(

module @overlay_packet_flow {
  aie.device(npu2_1col) {
    %home = aie.tile(0, 0)
    %t3 = aie.tile(0, 3)
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.packet_flow(5) {
      aie.packet_source<%home, DMA : 0>
      aie.packet_dest<%t3, DMA : 0>
    }
    aie.objectfifo @a(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
    }
  }
}
