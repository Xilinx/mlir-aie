//===- scratchpad_slots_per_command.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Each runtime sequence no aiex.run calls is a command, which a host runs with
// a scratchpad of its own. Its parameters, those of the sequences it runs and
// of their devices' cores, hold distinct state-table slots; parameters no one
// command reads together share one. The 32-entry state table bounds each
// command, not the module.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-scratchpad-parameters=output-params-file=%t.params %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=PARAMS < %t.params

// @shared, which both commands read, conflicts with the most parameters and
// takes slot 0. @a only `first` reads and @b only `second`: both take slot 1.
// `second`'s core parameter @c takes 2.

// CHECK: aiex.scratchpad_parameter @a : i32 {{.*}}state_table_idx = 1 : ui8}
// CHECK: aiex.scratchpad_parameter @b : i32 {{.*}}state_table_idx = 1 : ui8}
// CHECK: aiex.scratchpad_parameter @c : i32 {{.*}}state_table_idx = 2 : ui8}
// CHECK: aiex.scratchpad_parameter @shared : i32 {{.*}}state_table_idx = 0 : ui8}
// CHECK: aie.runtime_sequence @seq_a
// CHECK: aiex.npu.create_scratchpad {size = 12 : ui32}
// CHECK: aie.runtime_sequence @first
// CHECK-SAME: scratchpad_parameters = [@a, @shared]
// CHECK: aie.runtime_sequence @second
// CHECK-SAME: scratchpad_parameters = [@b, @c, @shared]

// PARAMS:      {{^}}4 3{{$}}
// PARAMS-NEXT: {{^}}a 1 i32 addr 0 960{{$}}
// PARAMS-NEXT: {{^}}b 1 i32 addr 0 960{{$}}
// PARAMS-NEXT: {{^}}c 2 i32 core - -{{$}}
// PARAMS-NEXT: {{^}}shared 0 i32 addr 0 960{{$}}
// PARAMS-NEXT: {{^}}0{{$}}
// PARAMS-NEXT: {{^}}2{{$}}
// PARAMS-NEXT: {{^}}main:first 2 a shared{{$}}
// PARAMS-NEXT: {{^}}main:second 3 b c shared{{$}}
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
