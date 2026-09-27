//===- scratchpad_size_parameter.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A size_parameter sets the extent of the third-innermost dimension (D2) of a
// shim BD each run. A shim BD has no D2 size field: its length ends D2. So the
// BD is written with length 0 and the firmware sets it to value * unit, where
// the unit is the two innermost dimensions in 32-bit words. The dimensions are
// written as for the static transfer, and are not folded into a linear one.
//
// A parameter only DMAs use is an `addr` parameter (stored raw, func_arg is
// the unit); one a core also reads is a `core` parameter, stored shifted left
// by 2, so func_arg is the unit / 4.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-substitute-shim-dma-allocations \
// RUN:   --aie-assign-runtime-sequence-bd-ids --aie-lower-scratchpad-parameters \
// RUN:   --aie-assign-buffer-addresses --aie-dma-tasks-to-npu --aie-dma-to-npu \
// RUN:   %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie-lower-scratchpad-parameters,aie.device(aie-normalize-dma-bd-dims))' \
// RUN:   %s | FileCheck %s --check-prefix=NORM

// Normalizing drops the size-1 dimension, which would make the size parameter
// name another dimension; a BD with one keeps its dimensions.
// NORM: aie.dma_bd({{.*}} sizes = [1, 16, 2, 32] strides = [0, 256, 64, 1]) {bd_id = 0 : i32, size_state_table_idx = 0 : ui8}

// CHECK-DAG: aiex.scratchpad_parameter @n : i32 {kind = 1 : i32, state_table_idx = 0 : ui8}
// CHECK-DAG: aiex.scratchpad_parameter @m : i32 {kind = 0 : i32, state_table_idx = 1 : ui8}
// The same words as the static transfer (length 512 and 256 words), with the
// length 0.
// CHECK-DAG: memref.global "private" constant @[[TASK:.*]] : memref<8xi32> = dense<[0, 0, 0, 16777216, -1071644641, 33554559, 0, 33554432]>
// CHECK-DAG: memref.global "private" constant @[[ND:.*]] : memref<8xi32> = dense<[0, 0, 0, 16777216, -1071644641, 33554559, 3146239, 33554432]>

// CHECK-LABEL: aie.runtime_sequence @seq
// The DMA-task BD 0 at 0x1D000: 2 x 32 bf16 = 32 words a unit.
// CHECK:      memref.get_global @[[TASK]]
// CHECK-NEXT: aiex.npu.blockwrite({{.*}}) {address = 118784 : ui32}
// CHECK:      aiex.npu.address_patch({{.*}}) {addr = 118788 : ui32, arg_idx = 0 : i32}
// CHECK-NEXT: aiex.npu.update_from_scratchpad<mul> {address = 118784 : ui32, func_arg = 32 : ui32, state_table_idx = 0 : ui8}
// BD 1, its size a core also reads: 32 words / 4.
// CHECK:      aiex.npu.blockwrite({{.*}}) {address = 118816 : ui32}
// CHECK:      aiex.npu.address_patch({{.*}}) {addr = 118820 : ui32, arg_idx = 1 : i32}
// CHECK-NEXT: aiex.npu.update_from_scratchpad<mul> {address = 118816 : ui32, func_arg = 8 : ui32, state_table_idx = 1 : ui8}
// The dma_memcpy_nd BD 2, an iteration dimension of 4: the length is per
// iteration, so each of the 4 moves the parameter's units.
// CHECK:      memref.get_global @[[ND]]
// CHECK-NEXT: aiex.npu.blockwrite({{.*}}) {address = 118848 : ui32}
// CHECK:      aiex.npu.address_patch({{.*}}) {addr = 118852 : ui32, arg_idx = 0 : i32}
// CHECK-NEXT: aiex.npu.update_from_scratchpad<mul> {address = 118848 : ui32, func_arg = 32 : ui32, state_table_idx = 0 : ui8}

module {
  aiex.scratchpad_parameter @n : i32
  aiex.scratchpad_parameter @m : i32
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.shim_dma_allocation @in (%t00, MM2S, 0)
    aie.shim_dma_allocation @out (%t00, S2MM, 0)
    %buf = aie.buffer(%t02) : memref<1xi32>
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %v = aiex.read_scratchpad_parameter @m : i32
      memref.store %v, %buf[%c0] : memref<1xi32>
      aie.end
    }
    aie.runtime_sequence @seq(%a : memref<4096xbf16>, %b : memref<4096xbf16>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<4096xbf16> offset = 0 len = 1024 sizes = [1, 16, 2, 32] strides = [0, 256, 64, 1]) {bd_id = 0 : i32, size_parameter = @n}
        aie.end
      }
      %u = aiex.dma_configure_task_for @out {
        aie.dma_bd(%b : memref<4096xbf16> offset = 0 len = 1024 sizes = [1, 16, 2, 32] strides = [0, 256, 64, 1]) {bd_id = 1 : i32, size_parameter = @m}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_start_task(%u)
      aiex.dma_await_task(%u)
      aiex.npu.dma_memcpy_nd(%a[0,0,0,0][4,8,2,32][1024,256,64,1]) {id = 2 : i64, metadata = @in, size_parameter = @n} : memref<4096xbf16>
    }
  }
}
