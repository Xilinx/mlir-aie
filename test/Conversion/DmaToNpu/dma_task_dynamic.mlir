//===- dma_task_dynamic.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Tests that aie-dma-tasks-to-npu lowers dynamic SSA operands on aie.dma_bd
// to NpuWriteBdOp (blockwrite template) + selective NpuWrite32Op overrides
// for dynamic BD words. The subsequent aie-dma-to-npu pass converts the
// NpuWriteBdOp to NpuBlockWriteOp.

// RUN: aie-opt --split-input-file --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu --verify-diagnostics %s | FileCheck %s

// All-static fast path: emits a single npu.writebd blockwrite.
// CHECK-LABEL: aie.device(npu1)
// CHECK: aiex.npu.writebd
// CHECK-NOT: aiex.npu.write32
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<128xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<128xi32>, 0, 32, [<size = 4, stride = 8>, <size = 8, stride = 1>]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Dynamic offset only: emits NpuWriteBdOp (static dims) + address_patch
// with dyn_arg_plus for the runtime offset. No write32 overrides needed
// since all sizes/strides are static.
// CHECK-LABEL: aie.device(npu1)
// CHECK: aiex.npu.writebd
// CHECK-NOT: aiex.npu.write32
// Dynamic offset: arg_plus is a runtime value (element offset * elem bytes).
// CHECK: %[[AP:.+]] = arith.muli
// CHECK: aiex.npu.address_patch(%[[AP]] : i32)
// CHECK: aiex.npu.push_queue
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<128xi32>, %M: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<128xi32>, 0, 32, [<size = 4, stride = 8>, <size = 8, stride = 1>]) dyn_offset(%M : i64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Fully dynamic sizes/strides: NpuWriteBdOp (blockwrite template with 0
// placeholders for dynamic fields) + npu.write32 overrides for the dynamic
// BD words (word[0]=bufLen, word[3..6]=sizes/strides), then address_patch
// and push_queue.
//
// Verify the arith computation chain that feeds the dynamic BD words. For
// memref<128xi32> on shim NOC (elemWidth==addrGran==32, so no scale/div):
//   - d0_stride: subi(stride0,1) / cmpi sgt stride0,0 / select  (stride>0 guard)
//   - d1_stride: subi(stride1,1) / cmpi sgt size1,1 / select    (size>1 guard)
//   - d2_stride: subi(stride2,1) / cmpi sgt size2,1 / select    (size>1 guard)
//   - iteration_stride: active = (size3>1 AND stride3>0) via arith.andi
//   - bufLen = hwD0Size * hwD1Size * hwD2Size via two muli ops
// All arith ops are emitted before aiex.npu.writebd; the dynamic BD words
// are then patched in by aiex.npu.write32 overrides.
// CHECK-LABEL: aie.device(npu1)
//
// d0_stride guard (stride0>0):
// CHECK: arith.subi
// CHECK: arith.cmpi sgt
// CHECK: arith.select
// d1_stride guard (size1>1):
// CHECK: arith.subi
// CHECK: arith.cmpi sgt
// CHECK: arith.select
// d2_stride guard (size2>1):
// CHECK: arith.subi
// CHECK: arith.cmpi sgt
// CHECK: arith.select
// iteration_stride: active = sizeGt1 AND strideGt0
// CHECK: arith.andi
// bufLen = d0_size * d1_size * d2_size:
// CHECK: arith.muli
// CHECK: arith.muli
//
// CHECK: aiex.npu.writebd
// CHECK: aiex.npu.write32
// CHECK: aiex.npu.address_patch
// CHECK: aiex.npu.write32
// CHECK: aiex.npu.sync
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<128xi32>, %M: i64, %S: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<128xi32>) dyn_offset(%M : i64) dyn_len(%M : i64) dyn_sizes(%M, %S, %M, %S : i64, i64, i64, i64) dyn_strides(%S, %M, %S, %M : i64, i64, i64, i64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Negative test: dynamic operands on MemTile should be rejected.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    aie.runtime_sequence(%arg0: memref<128xi32>, %M: i64) {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        // expected-error @+1 {{dynamic operands on aie.dma_bd are only supported on shim NOC tiles.}}
        aie.dma_bd(%arg0 : memref<128xi32>) dyn_len(%M : i64) dyn_sizes(%M, %M, %M, %M : i64, i64, i64, i64) dyn_strides(%M, %M, %M, %M : i64, i64, i64, i64) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
