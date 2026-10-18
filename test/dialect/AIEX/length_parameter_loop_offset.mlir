//===- length_parameter_loop_offset.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A length_parameter BD whose offset comes from a runtime-sequence loop. The
// offset is constant only after the loop is unrolled, so the parameter is
// bounded by the largest offset the loop reaches: at 2048, 64 static bf16 plus
// n units of 64 stay within 4096 elements up to n = 31. Each unrolled BD then
// patches its own address and takes the length from the scratchpad.

// RUN: aie-opt --aie-lower-scratchpad-parameters %s | FileCheck %s --check-prefix=BOUND
// RUN: aie-opt --aie-lower-scratchpad-parameters --aie-unroll-runtime-sequence-loops --canonicalize --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu %s | FileCheck %s

// BOUND: aiex.scratchpad_parameter @n : i32 {kind = 1 : i32, max_value = 31 : i32, min_value = 0 : i32

// CHECK-LABEL: aie.runtime_sequence @looped
// CHECK: aiex.npu.address_patch(%c0_i32 : i32) {addr = [[ADDR:[0-9]+]] : ui32, arg_idx = 0 : i32}
// CHECK: aiex.npu.update_from_scratchpad<mul> {{.*}}func_arg = 32 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.address_patch(%c4096_i32 : i32) {addr = [[ADDR]] : ui32, arg_idx = 0 : i32}
// CHECK: aiex.npu.update_from_scratchpad<mul> {{.*}}func_arg = 32 : ui32, state_table_idx = 0 : ui8}
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @looped(%arg0 : memref<4096xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c2048 = arith.constant 2048 : i32
    scf.for %i = %c0 to %c2 step %c1 {
      %ii = arith.index_cast %i : index to i32
      %off = arith.muli %ii, %c2048 : i32
      %task = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xbf16> offset = %off len = 64) {length_parameter = @n, length_unit = 64 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
    }
  }
}
