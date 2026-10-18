//===- scratchpad_parameter_kind_rerun.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host shifts a parameter's value by its kind, and the lowering drops the
// symbol uses the kind is derived from, so a second run keeps each kind. A
// repeat parameter two starts share is bounded by the smaller task count.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-scratchpad-parameters --aie-lower-scratchpad-parameters %s | FileCheck %s

// CHECK: aiex.scratchpad_parameter @n : i32 {kind = 2 : i32, max_value = 7 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
// CHECK: aiex.scratchpad_parameter @off : i32 {kind = 1 : i32, max_value = 960 : i32, min_value = 0 : i32, state_table_idx = 1 : ui8}
aiex.scratchpad_parameter @n : i32
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @d {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @s(%arg0 : memref<1024xi32>) {
    %a = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 64) {offset_parameter = @off}
      aie.end
    } {repeat_count = 15 : i32}
    %b = aiex.dma_configure_task(%t, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 64)
      aie.end
    } {repeat_count = 15 : i32}
    %c = aiex.dma_configure_task(%t, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 64)
      aie.end
    } {repeat_count = 7 : i32}
    aiex.dma_start_task(%a)
    aiex.dma_start_task(%b) {repeat_parameter = @n}
    aiex.dma_start_task(%c) {repeat_parameter = @n}
  }
}
