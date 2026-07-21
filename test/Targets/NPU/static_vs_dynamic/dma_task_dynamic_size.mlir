//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static-vs-dynamic equivalence for a genuinely runtime size on the DMA-task
// path (aiex.dma_configure_task / aie.dma_bd), the sibling of
// memcpy_nd_dynamic_size.mlir. This exercises the dma_task dynamic BD-word
// encoder (rewriteSingleBDDynamic): a runtime dma_bd size lowers to a
// zero-template writebd + write32 overrides, register-equivalent (not
// byte-equal) to the static baked writebd. Compiled with -DDYN_STRUCTURAL so
// the comparator replays both streams into register state.
//
// @task_static bakes the d2 size = 4; @task_dynamic takes it as %n.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// RUN: aie-opt --aie-substitute-shim-dma-allocations \
// RUN:   --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
// RUN:   --aie-dma-to-npu %s -o %t.d/lowered.mlir

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=task_static %t.d/lowered.mlir > %t.d/golden.hex

// RUN: aie-translate --aie-npu-to-cpp %t.d/lowered.mlir > %t.d/gen.h

// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_task_static \
// RUN:   -DDYN_FN=generate_txn_main_task_dynamic -DARGVAL=4 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)

    aie.runtime_sequence @task_static(%in: memref<8192xi32>) {
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<8192xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1]) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    aie.runtime_sequence @task_dynamic(%in: memref<8192xi32>, %n: i64) {
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<8192xi32> offset = 0 len = 1024 sizes = [1, %n, 8, 32] strides = [4096, 512, 32, 1]) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
