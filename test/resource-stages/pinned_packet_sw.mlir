//===- pinned_packet_sw.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// With packet-sw-objFifos every route asking for no packet header gets one, so
// fifos pinning one shim MM2S channel without a packet transport share it, as
// allocation lets them: on one logical shim (@one_tile), on a physical shim
// (@physical), and on two logical shims placed together (@two_tiles). Both
// placers place each and allocation gives both ends channel 0.

// RUN: aie-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{packet-sw-objFifos=true})' %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{packet-sw-objFifos=true placer=sa_placer sa-seed=1})' %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{packet-sw-objFifos=true placer=sa_placer sa-seed=2})' %s | FileCheck %s

// CHECK-LABEL: module @one_tile
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id =
// CHECK-DAG: aie.shim_dma_allocation @b__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id =
// CHECK-LABEL: module @physical
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id =
// CHECK-DAG: aie.shim_dma_allocation @b__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id =
// CHECK-LABEL: module @two_tiles
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id =
// CHECK-DAG: aie.shim_dma_allocation @b__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id =

module @one_tile {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %d = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%d}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}

// -----

module @physical {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %d = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%d}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}

// -----

module @two_tiles {
  aie.device(npu2_1col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %d = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%d}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}
