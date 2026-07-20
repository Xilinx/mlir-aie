//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// End-to-end proof that a rolled dynamic scf.if programs the same hardware as
// the static branch it selects. @rolled_if guards a single transfer with a
// runtime i1 condition (the form the static allocator rejects); it lowers
// through the dynamic BD pool to a C++ if. The comparator replays
// generate_txn_main_rolled_if(cond) and asserts it equals the TAKEN static
// oracle for cond=true and the NOT-TAKEN (empty) oracle for cond=false.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// Rolled dynamic scf.if -> generated C++.
// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s -o %t.d/rolled.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/rolled.mlir > %t.d/gen_rolled.h

// Static taken-branch oracle -> generated C++.
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
// RUN:   --aie-dma-to-npu %S/Inputs/rolled_if_static_taken.mlir -o %t.d/taken.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/taken.mlir > %t.d/gen_taken.h

// Static not-taken (empty) oracle -> generated C++.
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
// RUN:   --aie-dma-to-npu %S/Inputs/rolled_if_static_nottaken.mlir -o %t.d/nottaken.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/nottaken.mlir > %t.d/gen_nottaken.h

// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DROLLED_HDR='"%t.d/gen_rolled.h"' \
// RUN:   -DTAKEN_HDR='"%t.d/gen_taken.h"' \
// RUN:   -DNOTTAKEN_HDR='"%t.d/gen_nottaken.h"' \
// RUN:   %S/Inputs/rolled_if_compare.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @rolled_if(%arg0: memref<1024xi32>, %cond: i1) {
    scf.if %cond {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
