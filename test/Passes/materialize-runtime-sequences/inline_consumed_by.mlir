//===- inline_consumed_by.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// A cloned shim allocation's consumed_by names the clone of its output, which
// comes along even where the inlined sequence never names it. Two designs'
// allocations share their names, so one design's are renamed and each input's
// proof follows its own output, not the other design's.

// CHECK-LABEL: aie.device(npu2) {
// CHECK-DAG: aie.shim_dma_allocation @in(%{{.*}}, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 4>]}
// CHECK-DAG: aie.shim_dma_allocation @out(%{{.*}}, S2MM, 1)
// CHECK-DAG: aie.shim_dma_allocation @in_0(%{{.*}}, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out_0, input_bytes = 1, output_bytes = 1>]}
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
