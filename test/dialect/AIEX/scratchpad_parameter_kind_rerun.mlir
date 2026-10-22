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
// A second run also keeps each parameter's state-table slot, which the lowered
// transfers and core syncs read, and each command's parameters.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-lower-scratchpad-parameters --aie-lower-scratchpad-parameters %s | FileCheck %s

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

// -----

// CHECK: aiex.scratchpad_parameter @a : i32 {{.*}}state_table_idx = 1 : ui8}
// CHECK: aiex.scratchpad_parameter @b : i32 {{.*}}state_table_idx = 1 : ui8}
// CHECK: aiex.scratchpad_parameter @c : i32 {{.*}}state_table_idx = 2 : ui8}
// CHECK: aiex.scratchpad_parameter @shared : i32 {{.*}}state_table_idx = 0 : ui8}
// CHECK: aie.runtime_sequence @seq_a
// CHECK: offset_state_table_idx = 1 : ui8
// CHECK: aie.runtime_sequence @seq_b
// CHECK: aiex.npu.update_from_scratchpad {{.*}}state_table_idx = 2 : ui8
// CHECK: offset_state_table_idx = 1 : ui8
// CHECK: aie.runtime_sequence @seq_s
// CHECK: offset_state_table_idx = 0 : ui8
// CHECK: aie.runtime_sequence @first
// CHECK-SAME: scratchpad_parameters = [@a, @shared]
// CHECK: aie.runtime_sequence @second
// CHECK-SAME: scratchpad_parameters = [@b, @c, @shared]
module {
  aiex.scratchpad_parameter @a : i32
  aiex.scratchpad_parameter @b : i32
  aiex.scratchpad_parameter @c : i32
  aiex.scratchpad_parameter @shared : i32
  aie.device(npu2) @design_a {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in_a (%t, MM2S, 0)
    aie.runtime_sequence @seq_a(%in: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @in_a, offset_parameter = @a} : memref<1024xi32>
    }
  }
  aie.device(npu2) @design_b {
    %t = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.shim_dma_allocation @in_b (%t, MM2S, 0)
    aie.core(%t02) {
      %v = aiex.read_scratchpad_parameter @c : i32
      aie.end
    }
    aie.runtime_sequence @seq_b(%in: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @in_b, offset_parameter = @b} : memref<1024xi32>
    }
  }
  aie.device(npu2) @design_shared {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in_s (%t, MM2S, 0)
    aie.runtime_sequence @seq_s(%in: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @in_s, offset_parameter = @shared} : memref<1024xi32>
    }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @first(%in: memref<1024xi32>) {
      aiex.configure @design_a {
        aiex.run @seq_a(%in) : (memref<1024xi32>)
      }
      aiex.configure @design_shared {
        aiex.run @seq_s(%in) : (memref<1024xi32>)
      }
    }
    aie.runtime_sequence @second(%in: memref<1024xi32>) {
      aiex.configure @design_b {
        aiex.run @seq_b(%in) : (memref<1024xi32>)
      }
      aiex.configure @design_shared {
        aiex.run @seq_s(%in) : (memref<1024xi32>)
      }
    }
  }
}
