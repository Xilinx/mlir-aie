//===- scratchpad_unread_parameter.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A parameter no command reads gets no state-table slot, so it neither takes
// one of a command's 32 nor grows the scratchpad, and params.txt lists its
// slot as "-".
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-scratchpad-parameters=output-params-file=%t.params %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=PARAMS < %t.params

// CHECK: aiex.scratchpad_parameter @read : i32 {{.*}}state_table_idx = 0 : ui8}
// CHECK: aiex.scratchpad_parameter @unread : i32 {kind = 0 : i32}
// CHECK: aiex.npu.create_scratchpad {size = 4 : ui32}

// PARAMS:      {{^}}2 1{{$}}
// PARAMS-NEXT: {{^}}read 0 i32 core - -{{$}}
// PARAMS-NEXT: {{^}}unread - i32 core - -{{$}}
// PARAMS-NEXT: {{^}}0{{$}}
// PARAMS-NEXT: {{^}}1{{$}}
// PARAMS-NEXT: {{^}}main:seq 1 read{{$}}
module {
  aiex.scratchpad_parameter @read : i32
  aiex.scratchpad_parameter @unread : i32
  aie.device(npu2) @main {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %v = aiex.read_scratchpad_parameter @read : i32
      aie.end
    }
    aie.runtime_sequence @seq(%arg0 : memref<16xi32>) {
    }
  }
}
