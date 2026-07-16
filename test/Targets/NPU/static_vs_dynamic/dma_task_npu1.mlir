//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// npu1 counterpart of dma_task.mlir (the DMA-task path). Device info is baked
// into the TXN header words, so the static ≡ dynamic equivalence is checked
// per device generation; this file covers npu1 while dma_task.mlir covers
// npu2. See that file and README.md for details.
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
// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_task_static \
// RUN:   -DDYN_FN=generate_txn_main_task_dynamic -DARGVAL=4096 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %rtp = aie.buffer(%tile_0_2) {sym_name = "rtp", address = 49152 : i32} : memref<16xi32>
    aie.shim_dma_allocation @of_in  (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @of_out (%tile_0_0, S2MM, 0)

    aie.runtime_sequence @task_static(%in: memref<8192xi32>, %out: memref<8192xi32>) {
      %c4096 = arith.constant 4096 : i32
      %c4097 = arith.constant 4097 : i32
      aiex.npu.rtp_write(@rtp, 0, %c4096) : i32
      aiex.npu.rtp_write(@rtp, 4, %c4097) : i32
      %tout = aiex.dma_configure_task_for @of_out {
        aie.dma_bd(%out : memref<8192xi32> offset = 0 len = 2048 sizes = [2, 4, 8, 64] strides = [2048, 512, 64, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tout)
      %tin = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<8192xi32> offset = 0 len = 4096 sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tin)
      aiex.dma_await_task(%tout)
      aiex.dma_await_task(%tin)
    }

    aie.runtime_sequence @task_dynamic(%in: memref<8192xi32>, %out: memref<8192xi32>, %n: i32) {
      %c1 = arith.constant 1 : i32
      %np1 = arith.addi %n, %c1 : i32
      aiex.npu.rtp_write(@rtp, 0, %n) : i32
      aiex.npu.rtp_write(@rtp, 4, %np1) : i32
      %tout = aiex.dma_configure_task_for @of_out {
        aie.dma_bd(%out : memref<8192xi32> offset = 0 len = 2048 sizes = [2, 4, 8, 64] strides = [2048, 512, 64, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tout)
      %tin = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<8192xi32> offset = 0 len = 4096 sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tin)
      aiex.dma_await_task(%tout)
      aiex.dma_await_task(%tin)
    }
  }
}
