//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static-vs-dynamic equivalence for a runtime *repeat_count* (the outer/d3
// dimension) on dma_memcpy_nd -- milestone #3222 PR 6's "dynamic repeat_count"
// case. A runtime outer size drives both the BD iteration word AND the queue
// push command word (bd_id | repeat<<16 | issue<<31), which the push-queue
// lowering now builds with arith instead of rejecting. Register-replay
// (-DDYN_STRUCTURAL) proves the runtime and static-baked streams program the
// same registers.
//
// @rep_static bakes the outer size = 4 (repeat_count 3); @rep_dynamic takes it
// as %n.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// RUN: aie-opt --aie-dma-to-npu %s -o %t.d/lowered.mlir

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=rep_static %t.d/lowered.mlir > %t.d/golden.hex

// RUN: aie-translate --aie-npu-to-cpp %t.d/lowered.mlir > %t.d/gen.h

// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_rep_static \
// RUN:   -DDYN_FN=generate_txn_main_rep_dynamic -DARGVAL=4 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)

    // Contiguous inner block (linear), runtime outer repeat count.
    aie.runtime_sequence @rep_static(%in: memref<8192xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][4, 1, 8, 32][256, 0, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }

    aie.runtime_sequence @rep_dynamic(%in: memref<8192xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][%n, 1, 8, 32][256, 0, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }
  }
}
