//===- inline_consumed_by.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --split-input-file %s | FileCheck %s
// RUN: split-file %s %t
// RUN: aie-opt --aie-materialize-runtime-sequences --aie-unroll-runtime-sequence-loops --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids="warn-unreleased-transfers=false warn-unsafe-bd-reuse=false" --split-input-file %t/tasks.mlir > /dev/null

// A cloned shim allocation's consumed_by names the clone of its output, which
// comes along even where the inlined sequence never names it. Two designs'
// allocations share their names, so one design's are renamed and each input's
// proof follows its own output, not the other design's.

// CHECK-LABEL: aie.device(npu2) {
// CHECK-DAG: aie.shim_dma_allocation @in(%{{.*}}, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 4, device = @b>]}
// CHECK-DAG: aie.shim_dma_allocation @out(%{{.*}}, S2MM, 1)
// CHECK-DAG: aie.shim_dma_allocation @in_0(%{{.*}}, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out_0, input_bytes = 1, output_bytes = 1, device = @a>]}
// CHECK-DAG: aie.shim_dma_allocation @out_0(%{{.*}}, S2MM, 0)
// CHECK: aiex.npu.load_pdi {device_ref = @a}
// CHECK: aiex.dma_configure_task_for @in_0 {
// CHECK: aiex.npu.load_pdi {device_ref = @b}
// CHECK: aiex.dma_configure_task_for @in {
// CHECK-LABEL: aie.device(npu2) @a
module {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      aiex.configure @a {
        aiex.run @seq(%arg0) : (memref<64xi32>)
      }
      aiex.configure @b {
        aiex.run @seq(%arg0) : (memref<64xi32>)
      }
    }
  }

  aie.device(npu2) @a {
    %tile00 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile00, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile00, S2MM, 0)
    aie.runtime_sequence @seq(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }

  aie.device(npu2) @b {
    %tile00 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile00, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 4>]}
    aie.shim_dma_allocation @out(%tile00, S2MM, 1)
    aie.runtime_sequence @seq(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

//--- tasks.mlir

// CHECK-LABEL: aie.runtime_sequence @calls
// CHECK: %[[FIRST_ALLOC:.*]] = aiex.dma_configure_task_for
// CHECK: %[[FIRST_TILE:.*]] = aiex.dma_configure_task(
// CHECK: aiex.dma_free_task(%[[FIRST_TILE]])
// CHECK-NEXT: aiex.dma_free_task(%[[FIRST_ALLOC]])
// CHECK-NEXT: %[[SECOND_ALLOC:.*]] = aiex.dma_configure_task_for
// CHECK: %[[SECOND_TILE:.*]] = aiex.dma_configure_task(
// CHECK: aiex.dma_free_task(%[[SECOND_TILE]])
// CHECK-NEXT: aiex.dma_free_task(%[[SECOND_ALLOC]])
// CHECK-NEXT: }
module {
  aie.device(npu2) @main {
    aie.runtime_sequence @calls(%buffer: memref<16xi32>) {
      aiex.configure @worker {
        aiex.run @tasks(%buffer) : (memref<16xi32>)
        aiex.run @tasks(%buffer) : (memref<16xi32>)
      }
    }
  }

  aie.device(npu2) @worker {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @input(%shim, MM2S, 0)
    aie.runtime_sequence @tasks(%buffer: memref<16xi32>) {
      %by_alloc = aiex.dma_configure_task_for @input {
        aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 8)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%by_alloc)
      aiex.dma_await_task(%by_alloc)
      %by_tile = aiex.dma_configure_task(%shim, MM2S, 1) {
        aie.dma_bd(%buffer : memref<16xi32> offset = 8 len = 8)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%by_tile)
      aiex.dma_await_task(%by_tile)
      aiex.dma_free_task(%by_tile)
    }
  }
}

// -----

// CHECK-LABEL: aie.runtime_sequence @forwarded
// CHECK: scf.for
// CHECK: aiex.dma_free_task
// CHECK-NOT: aiex.dma_free_task
// CHECK-LABEL: aie.device(npu2) @worker
module {
  aie.device(npu2) @main {
    aie.runtime_sequence @forwarded(%buffer: memref<16xi32>) {
      aiex.configure @worker {
        aiex.run @tasks(%buffer) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu2) @worker {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @input(%shim, MM2S, 0)
    aie.runtime_sequence @tasks(%buffer: memref<16xi32>) {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %task = aiex.dma_configure_task_for @input {
        aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.npu.dma_wait {symbol = @input}
      %result = scf.for %iteration = %zero to %one step %one iter_args(%forwarded = %task) -> (index) {
        aiex.dma_free_task(%forwarded)
        scf.yield %forwarded : index
      }
    }
  }
}
