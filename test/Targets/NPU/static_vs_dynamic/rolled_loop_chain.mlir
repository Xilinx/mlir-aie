//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// rolled_loop.mlir with every task a 2-BD next_bd chain. The rolled loop pops
// one pool id per BD and links each BD's next_bd to its successor's runtime id;
// Inputs/rolled_loop_chain_static2.mlir is the hand-unrolled 2-iteration
// oracle, whose chains the static allocator pins. rolled_loop_compare.cpp
// replays generate_txn_main_rolled(2) and generate_txn_main_static2() into
// register maps and asserts equality, so the runtime next_bd words must match
// the static ones bit for bit.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// Queue-depth enforcement is off on both sides for the reason rolled_loop.mlir
// gives: only BD register programming is compared.
// RUN: aie-opt --aie-lower-dynamic-bd-pool='enforce-queue-depth=false' --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu='enforce-queue-depth=false' %s -o %t.d/rolled.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/rolled.mlir > %t.d/gen_rolled.h

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='enforce-queue-depth=false' --aie-dma-tasks-to-npu \
// RUN:   --aie-dma-to-npu='enforce-queue-depth=false' %S/Inputs/rolled_loop_chain_static2.mlir -o %t.d/static2.mlir
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
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 512 sizes = [1, 2, 8, 32] strides = [4096, 256, 32, 1])
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 512 sizes = [1, 2, 8, 32] strides = [4096, 256, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 512 sizes = [1, 2, 8, 32] strides = [4096, 256, 32, 1])
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 512 sizes = [1, 2, 8, 32] strides = [4096, 256, 32, 1])
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
