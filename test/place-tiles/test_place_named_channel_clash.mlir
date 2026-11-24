//===- test_place_named_channel_clash.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// npu1_1col has one mem tile. Two logical mem tiles whose DMA programs both
// start MM2S 0 cannot share it: one chain would never run behind the other's
// endless one. Both placers refuse, naming the clash. Two whose flows leave
// by different channels share it.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=2' %s | FileCheck %s

module @one_col_named_programs {
  aie.device(npu1_1col) {
    // expected-note@+1 {{the tile op naming MM2S channel 0}}
    %mem_a = aie.logical_tile<MemTile>(?, ?)
    // expected-error@+1 {{names MM2S channel 0, as a tile op on (0, 1) does, and each MemTile with room for it holds a tile op naming one of its channels; tile ops whose DMA programs or flows name one channel cannot share a physical tile, so give one of them another channel}}
    %mem_b = aie.logical_tile<MemTile>(?, ?)
    %core_a = aie.tile(0, 2)
    %core_b = aie.tile(0, 3)
    %buf_a = aie.buffer(%mem_a) {sym_name = "buf_a"} : memref<16xi32>
    %buf_b = aie.buffer(%mem_b) {sym_name = "buf_b"} : memref<16xi32>
    aie.flow(%mem_a, DMA : 0, %core_a, DMA : 0)
    aie.flow(%mem_b, DMA : 0, %core_b, DMA : 0)
    %dma_a = aie.memtile_dma(%mem_a) {
      %s = aie.dma_start(MM2S, 0, ^b, ^end)
    ^b:
      aie.dma_bd(%buf_a : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^b
    ^end:
      aie.end
    }
    %dma_b = aie.memtile_dma(%mem_b) {
      %s = aie.dma_start(MM2S, 0, ^b, ^end)
    ^b:
      aie.dma_bd(%buf_b : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.next_bd ^b
    ^end:
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: @one_col_distinct
// CHECK: %[[M:.*]] = aie.tile(0, 1)
// CHECK-DAG: aie.flow(%[[M]], DMA : 0, %{{.*}}, DMA : 0)
// CHECK-DAG: aie.flow(%[[M]], DMA : 1, %{{.*}}, DMA : 0)
module @one_col_distinct {
  aie.device(npu1_1col) {
    %mem_a = aie.logical_tile<MemTile>(?, ?)
    %mem_b = aie.logical_tile<MemTile>(?, ?)
    %core_a = aie.tile(0, 2)
    %core_b = aie.tile(0, 3)
    aie.flow(%mem_a, DMA : 0, %core_a, DMA : 0)
    aie.flow(%mem_b, DMA : 1, %core_b, DMA : 0)
  }
}
