//===- test_place_shim_sharing.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles --aie-objectFifo-stateful-transform %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s --check-prefixes=CHECK,SA
// RUN: sed -e 's/in_a/in_z/g' -e 's/in_c/in_a/g' %s > %t.renamed.mlir
// RUN: aie-opt --split-input-file --aie-place-tiles --aie-objectFifo-stateful-transform %t.renamed.mlir | FileCheck %t.renamed.mlir

// The placer counts a shim tile's channels the way objectFIFO allocation
// shares them: once the tile runs out, ends never in flight together count
// once. Three members of a pack, one runtime sequence each, fill three ends on
// the one shim tile npu2_1col has, which sends on two channels; two of them
// take turns on one.
// SA-NOT:      warning
// CHECK-LABEL: module @pack
// CHECK-DAG:   aie.shim_dma_allocation @in_a__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:   aie.shim_dma_allocation @in_b__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:   aie.shim_dma_allocation @in_c__shim_alloc(%{{.*}}, MM2S, 1)
module @pack {
  aie.device(npu2_1col) {
    %s_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %s_b = aie.logical_tile<ShimNOCTile>(?, ?)
    %s_c = aie.logical_tile<ShimNOCTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in_a(%s_a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s_b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_c(%s_c, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_a(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @in_c(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_c(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run_a(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<16xi32>
    }
    aie.runtime_sequence @run_b(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64} : memref<16xi32>
    }
    aie.runtime_sequence @run_c(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_c, id = 0 : i64} : memref<16xi32>
    }
  }
}

// -----

// Two ends fit the tile, so nothing is shared and both keep a channel each.
// CHECK-LABEL: module @fits
// CHECK-DAG:   aie.shim_dma_allocation @in_a__shim_alloc(%{{.*}}, MM2S, 0)
// CHECK-DAG:   aie.shim_dma_allocation @in_b__shim_alloc(%{{.*}}, MM2S, 1)
module @fits {
  aie.device(npu2_1col) {
    %s_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %s_b = aie.logical_tile<ShimNOCTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in_a(%s_a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s_b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_a(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run_a(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<16xi32>
    }
    aie.runtime_sequence @run_b(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64} : memref<16xi32>
    }
  }
}
