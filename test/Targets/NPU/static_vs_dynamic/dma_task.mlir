//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static-vs-dynamic TXN equivalence for the DMA-task path
// (dma_configure_task_for / dma_start_task / dma_await_task, carrying the SSA
// bd_id introduced in #3225).
//
// This path lowers differently from dma_memcpy_nd: dma-tasks-to-npu first
// produces npu.push_queue / npu.write_bd (and an npu.sync per awaited token),
// which dma-to-npu then lowers to the same terminal write32 / blockwrite /
// sync / address_patch ops. This test proves that convergence end to end -
// nothing else in the tree exercises a DMA-task sequence all the way to a TXN
// stream - and that the C++ builder is byte-identical to the binary emitter
// on it, for both a static and a runtime-arg sequence.
//
// The await steps mean issue_token=true is required on the configures, so this
// is also the path that exercises the sync op in the EmitC target. Two BD
// shapes (repeat count in the highest dim; length is the product of the lower
// three) cover more than one BD encoding.
//
// To add another size, see README.md in this directory.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// Lower the DMA-task ops to terminal npu ops. Unlike the dma_memcpy_nd path
// this needs the shim substitution + BD-id assignment + dma-tasks-to-npu
// stages before the shared dma-to-npu step.
// RUN: aie-opt --aie-substitute-shim-dma-allocations \
// RUN:   --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
// RUN:   --aie-dma-to-npu %s -o %t.d/lowered.mlir

// Golden word stream from the production binary emitter.
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=task_static %t.d/lowered.mlir > %t.d/golden.hex

// One generated header holds both generate_txn_main_task_static/_dynamic.
// RUN: aie-translate --aie-npu-to-cpp %t.d/lowered.mlir > %t.d/gen.h

// Host-compile and run the three-way comparator.
// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_task_static \
// RUN:   -DDYN_FN=generate_txn_main_task_dynamic -DARGVAL=4096 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

module {
  aie.device(npu2) {
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
