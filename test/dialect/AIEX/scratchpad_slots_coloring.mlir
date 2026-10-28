//===- scratchpad_slots_coloring.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Three commands: c1 reads @x0 and @z, c2 the 31 @y and @z, c3 @w0 and the 31
// @y. They fit 32 state-table slots only if @z shares @w0's and @x0 shares one
// of the @y's. Taking the parameters in declaration order, @x0 and @w0 first,
// fills all 32 before @z; the slots are assigned most constrained first, so
// the declaration order does not decide whether the module fits.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-scratchpad-parameters %s | FileCheck %s

// CHECK: aiex.scratchpad_parameter @x0 : i32 {{.*}}state_table_idx = 0 : ui8}
// CHECK: aiex.scratchpad_parameter @w0 : i32 {{.*}}state_table_idx = 31 : ui8}
// CHECK: aiex.scratchpad_parameter @y0 : i32 {{.*}}state_table_idx = 0 : ui8}
// CHECK: aiex.scratchpad_parameter @y30 : i32 {{.*}}state_table_idx = 30 : ui8}
// CHECK: aiex.scratchpad_parameter @z : i32 {{.*}}state_table_idx = 31 : ui8}
// CHECK: aiex.npu.create_scratchpad {size = 128 : ui32}
module {
  aiex.scratchpad_parameter @x0 : i32
  aiex.scratchpad_parameter @w0 : i32
  aiex.scratchpad_parameter @y0 : i32
  aiex.scratchpad_parameter @y1 : i32
  aiex.scratchpad_parameter @y2 : i32
  aiex.scratchpad_parameter @y3 : i32
  aiex.scratchpad_parameter @y4 : i32
  aiex.scratchpad_parameter @y5 : i32
  aiex.scratchpad_parameter @y6 : i32
  aiex.scratchpad_parameter @y7 : i32
  aiex.scratchpad_parameter @y8 : i32
  aiex.scratchpad_parameter @y9 : i32
  aiex.scratchpad_parameter @y10 : i32
  aiex.scratchpad_parameter @y11 : i32
  aiex.scratchpad_parameter @y12 : i32
  aiex.scratchpad_parameter @y13 : i32
  aiex.scratchpad_parameter @y14 : i32
  aiex.scratchpad_parameter @y15 : i32
  aiex.scratchpad_parameter @y16 : i32
  aiex.scratchpad_parameter @y17 : i32
  aiex.scratchpad_parameter @y18 : i32
  aiex.scratchpad_parameter @y19 : i32
  aiex.scratchpad_parameter @y20 : i32
  aiex.scratchpad_parameter @y21 : i32
  aiex.scratchpad_parameter @y22 : i32
  aiex.scratchpad_parameter @y23 : i32
  aiex.scratchpad_parameter @y24 : i32
  aiex.scratchpad_parameter @y25 : i32
  aiex.scratchpad_parameter @y26 : i32
  aiex.scratchpad_parameter @y27 : i32
  aiex.scratchpad_parameter @y28 : i32
  aiex.scratchpad_parameter @y29 : i32
  aiex.scratchpad_parameter @y30 : i32
  aiex.scratchpad_parameter @z : i32
  aie.device(npu2) @dx {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      %x0 = aiex.read_scratchpad_parameter @x0 : i32
      aie.end
    }
    aie.runtime_sequence @seq_dx(%arg0 : memref<16xi32>) {
    }
  }
  aie.device(npu2) @dw {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      %w0 = aiex.read_scratchpad_parameter @w0 : i32
      aie.end
    }
    aie.runtime_sequence @seq_dw(%arg0 : memref<16xi32>) {
    }
  }
  aie.device(npu2) @dy {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      %y0 = aiex.read_scratchpad_parameter @y0 : i32
      %y1 = aiex.read_scratchpad_parameter @y1 : i32
      %y2 = aiex.read_scratchpad_parameter @y2 : i32
      %y3 = aiex.read_scratchpad_parameter @y3 : i32
      %y4 = aiex.read_scratchpad_parameter @y4 : i32
      %y5 = aiex.read_scratchpad_parameter @y5 : i32
      %y6 = aiex.read_scratchpad_parameter @y6 : i32
      %y7 = aiex.read_scratchpad_parameter @y7 : i32
      %y8 = aiex.read_scratchpad_parameter @y8 : i32
      %y9 = aiex.read_scratchpad_parameter @y9 : i32
      %y10 = aiex.read_scratchpad_parameter @y10 : i32
      %y11 = aiex.read_scratchpad_parameter @y11 : i32
      %y12 = aiex.read_scratchpad_parameter @y12 : i32
      %y13 = aiex.read_scratchpad_parameter @y13 : i32
      %y14 = aiex.read_scratchpad_parameter @y14 : i32
      %y15 = aiex.read_scratchpad_parameter @y15 : i32
      %y16 = aiex.read_scratchpad_parameter @y16 : i32
      %y17 = aiex.read_scratchpad_parameter @y17 : i32
      %y18 = aiex.read_scratchpad_parameter @y18 : i32
      %y19 = aiex.read_scratchpad_parameter @y19 : i32
      %y20 = aiex.read_scratchpad_parameter @y20 : i32
      %y21 = aiex.read_scratchpad_parameter @y21 : i32
      %y22 = aiex.read_scratchpad_parameter @y22 : i32
      %y23 = aiex.read_scratchpad_parameter @y23 : i32
      %y24 = aiex.read_scratchpad_parameter @y24 : i32
      %y25 = aiex.read_scratchpad_parameter @y25 : i32
      %y26 = aiex.read_scratchpad_parameter @y26 : i32
      %y27 = aiex.read_scratchpad_parameter @y27 : i32
      %y28 = aiex.read_scratchpad_parameter @y28 : i32
      %y29 = aiex.read_scratchpad_parameter @y29 : i32
      %y30 = aiex.read_scratchpad_parameter @y30 : i32
      aie.end
    }
    aie.runtime_sequence @seq_dy(%arg0 : memref<16xi32>) {
    }
  }
  aie.device(npu2) @dz {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      %z = aiex.read_scratchpad_parameter @z : i32
      aie.end
    }
    aie.runtime_sequence @seq_dz(%arg0 : memref<16xi32>) {
    }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @c1(%arg0 : memref<16xi32>) {
      aiex.configure @dx {
        aiex.run @seq_dx(%arg0) : (memref<16xi32>)
      }
      aiex.configure @dz {
        aiex.run @seq_dz(%arg0) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @c2(%arg0 : memref<16xi32>) {
      aiex.configure @dy {
        aiex.run @seq_dy(%arg0) : (memref<16xi32>)
      }
      aiex.configure @dz {
        aiex.run @seq_dz(%arg0) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @c3(%arg0 : memref<16xi32>) {
      aiex.configure @dw {
        aiex.run @seq_dw(%arg0) : (memref<16xi32>)
      }
      aiex.configure @dy {
        aiex.run @seq_dy(%arg0) : (memref<16xi32>)
      }
    }
  }
}
