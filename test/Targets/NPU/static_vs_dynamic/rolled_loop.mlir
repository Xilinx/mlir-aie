//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// End-to-end proof that a ROLLED dynamic loop programs the same hardware as N
// unrolled static configures. @rolled is a runtime-bound ping-pong (the form
// the static allocator rejects); it lowers through the dynamic BD pool to a C++
// for-loop. Inputs/rolled_loop_static2.mlir is the hand-unrolled 2-iteration
// static oracle. The comparator replays generate_txn_main_rolled(2) and the
// static generate_txn_main_static2() into register maps and asserts equality.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// Rolled dynamic loop -> generated C++.
// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s -o %t.d/rolled.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/rolled.mlir > %t.d/gen_rolled.h

// Static hand-unrolled n=2 oracle -> generated C++.
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
// RUN:   --aie-dma-to-npu %S/Inputs/rolled_loop_static2.mlir -o %t.d/static2.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/static2.mlir > %t.d/gen_static2.h

// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DROLLED_HDR='"%t.d/gen_rolled.h"' \
// RUN:   -DSTATIC_HDR='"%t.d/gen_static2.h"' \
// RUN:   %S/Inputs/rolled_loop_compare.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @rolled(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}
