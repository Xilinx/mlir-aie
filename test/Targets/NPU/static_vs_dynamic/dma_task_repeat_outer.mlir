//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static-vs-dynamic equivalence for a pure-REPEAT outer dimension: the highest
// dimension has size > 1 and stride 0, so the BD wraps every iteration and the
// repeat is carried by the queue push's repeat_count -- NOT by the BD's
// iteration_size/iteration_stride word, which must be 0.
//
// This is the pattern the whole-array GEMM's A/B input BDs use (sizes
// [16, 8, 256, 64] strides [0, 64, 512, 1]) and the case the other
// static_vs_dynamic tests did not cover: they only drive a runtime INNER size.
// The dynamic BD-word encoder used to pack iteration_size unconditionally
// (BdLowering.cpp), giving word6 a nonzero value where the static path emits 0
// -- corrupting every input tile fetch. This test locks in the fix by requiring
// the dynamic stream to stay register-equivalent to the static baked one when
// the outer (repeat) size is runtime.
//
// @task_static bakes the outer repeat size = 16; @task_dynamic takes it as %n.
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
// RUN:   -DDYN_FN=generate_txn_main_task_dynamic -DARGVAL=16 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)

    aie.runtime_sequence @task_static(%in: memref<262144xi16>) {
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<262144xi16> offset = 0 len = 131072 sizes = [16, 8, 256, 64] strides = [0, 64, 512, 1]) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }

    aie.runtime_sequence @task_dynamic(%in: memref<262144xi16>, %n: i64) {
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<262144xi16> offset = 0 len = 131072 sizes = [%n, 8, 256, 64] strides = [0, 64, 512, 1]) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
