//===- dma_task_dynamic_memtile_core.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The mem tile and core tile siblings of dma_task_dynamic_size.mlir: a runtime
// size or offset on a non-shim BD must program the same registers as the
// static baked writebd. The mem tile cases use odd channel 1, whose BDs start
// at 24, and the offset cases go through the word-addressed buffer field.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
// RUN:   --aie-dma-to-npu %s -o %t.d/lowered.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/lowered.mlir > %t.d/gen.h

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=mem_size_static %t.d/lowered.mlir > %t.d/mem_size.hex
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_mem_size_static \
// RUN:   -DDYN_FN=generate_txn_main_mem_size_dynamic -DARGVAL=4 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/mem_size.exe
// RUN: %t.d/mem_size.exe %t.d/mem_size.hex

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=mem_offset_static %t.d/lowered.mlir > %t.d/mem_offset.hex
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_mem_offset_static \
// RUN:   -DDYN_FN=generate_txn_main_mem_offset_dynamic -DARGVAL=256 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/mem_offset.exe
// RUN: %t.d/mem_offset.exe %t.d/mem_offset.hex

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=core_size_static %t.d/lowered.mlir > %t.d/core_size.hex
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_core_size_static \
// RUN:   -DDYN_FN=generate_txn_main_core_size_dynamic -DARGVAL=4 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/core_size.exe
// RUN: %t.d/core_size.exe %t.d/core_size.hex

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=core_offset_static %t.d/lowered.mlir > %t.d/core_offset.hex
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_core_offset_static \
// RUN:   -DDYN_FN=generate_txn_main_core_offset_dynamic -DARGVAL=64 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/core_offset.exe
// RUN: %t.d/core_offset.exe %t.d/core_offset.hex

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %mem_buf = aie.buffer(%tile_0_1) {address = 4096 : i32} : memref<4096xi32>
    %core_buf = aie.buffer(%tile_0_2) {address = 4096 : i32} : memref<1024xi32>

    aie.runtime_sequence @mem_size_static() {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 1) {
        aie.dma_bd(%mem_buf : memref<4096xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [0, 512, 32, 1]) {bd_id = 24 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    aie.runtime_sequence @mem_size_dynamic(%n: i64) {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 1) {
        aie.dma_bd(%mem_buf : memref<4096xi32> offset = 0 len = 1024 sizes = [1, %n, 8, 32] strides = [0, 512, 32, 1]) {bd_id = 24 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    aie.runtime_sequence @mem_offset_static() {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 1) {
        aie.dma_bd(%mem_buf : memref<4096xi32> offset = 256 len = 512) {bd_id = 24 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    aie.runtime_sequence @mem_offset_dynamic(%off: i32) {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 1) {
        aie.dma_bd(%mem_buf : memref<4096xi32> offset = %off len = 512) {bd_id = 24 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    aie.runtime_sequence @core_size_static() {
      %t = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
        aie.dma_bd(%core_buf : memref<1024xi32> offset = 0 len = 512 sizes = [1, 4, 4, 32] strides = [0, 128, 32, 1]) {bd_id = 3 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    aie.runtime_sequence @core_size_dynamic(%n: i64) {
      %t = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
        aie.dma_bd(%core_buf : memref<1024xi32> offset = 0 len = 512 sizes = [1, %n, 4, 32] strides = [0, 128, 32, 1]) {bd_id = 3 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    aie.runtime_sequence @core_offset_static() {
      %t = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
        aie.dma_bd(%core_buf : memref<1024xi32> offset = 64 len = 256) {bd_id = 3 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    aie.runtime_sequence @core_offset_dynamic(%off: i32) {
      %t = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
        aie.dma_bd(%core_buf : memref<1024xi32> offset = %off len = 256) {bd_id = 3 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
