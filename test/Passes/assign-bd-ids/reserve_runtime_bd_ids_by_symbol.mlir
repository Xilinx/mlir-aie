//===- reserve_runtime_bd_ids_by_symbol.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-reserve-runtime-bd-ids --aie-assign-bd-ids \
// RUN:   --aie-substitute-shim-dma-allocations \
// RUN:   --aie-assign-runtime-sequence-bd-ids --split-input-file %s | FileCheck %s

// A task that reaches its tile through a shim DMA allocation pins its id as a
// task on the tile does: the static BD takes the next one.

// CHECK-LABEL: @task_for
// CHECK: aie.tile(0, 0) {aiex.reserved_bd_ids = array<i32: 0>}
// CHECK: aie.dma_bd(%{{.*}} : memref<8xi32>) {bd_id = 1 : i32
module @task_for {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    %buf = aie.external_buffer {sym_name = "b"} : memref<8xi32>
    aie.shim_dma_allocation @in(%t, S2MM, 1)
    aie.shim_dma(%t) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buf : memref<8xi32>)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.runtime_sequence(%a: memref<8xi32>) {
      %task = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
    }
  }
}

// -----

// So does a dma_memcpy_nd's id.

// CHECK-LABEL: @copy
// CHECK: aie.tile(0, 0) {aiex.reserved_bd_ids = array<i32: 0>}
// CHECK: aie.dma_bd(%{{.*}} : memref<8xi32>) {bd_id = 1 : i32
module @copy {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    %buf = aie.external_buffer {sym_name = "b"} : memref<8xi32>
    aie.shim_dma_allocation @in(%t, S2MM, 1)
    aie.shim_dma(%t) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buf : memref<8xi32>)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.runtime_sequence(%a: memref<8xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, metadata = @in, issue_token = true} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @in}
    }
  }
}
