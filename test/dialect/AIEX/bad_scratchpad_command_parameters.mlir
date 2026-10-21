//===- bad_scratchpad_command_parameters.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A command's scratchpad parameters each take one of the command processor's
// 32 state-table entries, so one command reading 33 is refused. Two
// parameters one command reads may not keep the same entry from before, nor
// one keep an entry past the 32. Three commands reading 22 each, @a with @b,
// @b with @c and @a with @c, need 33 distinct entries between them, which no
// assignment has.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-lower-scratchpad-parameters %s 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: error: runtime sequence 'main:seq' reads 33 parameters but its scratchpad holds at most 32
module {
  aiex.scratchpad_parameter @p0 : i32
  aiex.scratchpad_parameter @p1 : i32
  aiex.scratchpad_parameter @p2 : i32
  aiex.scratchpad_parameter @p3 : i32
  aiex.scratchpad_parameter @p4 : i32
  aiex.scratchpad_parameter @p5 : i32
  aiex.scratchpad_parameter @p6 : i32
  aiex.scratchpad_parameter @p7 : i32
  aiex.scratchpad_parameter @p8 : i32
  aiex.scratchpad_parameter @p9 : i32
  aiex.scratchpad_parameter @p10 : i32
  aiex.scratchpad_parameter @p11 : i32
  aiex.scratchpad_parameter @p12 : i32
  aiex.scratchpad_parameter @p13 : i32
  aiex.scratchpad_parameter @p14 : i32
  aiex.scratchpad_parameter @p15 : i32
  aiex.scratchpad_parameter @p16 : i32
  aiex.scratchpad_parameter @p17 : i32
  aiex.scratchpad_parameter @p18 : i32
  aiex.scratchpad_parameter @p19 : i32
  aiex.scratchpad_parameter @p20 : i32
  aiex.scratchpad_parameter @p21 : i32
  aiex.scratchpad_parameter @p22 : i32
  aiex.scratchpad_parameter @p23 : i32
  aiex.scratchpad_parameter @p24 : i32
  aiex.scratchpad_parameter @p25 : i32
  aiex.scratchpad_parameter @p26 : i32
  aiex.scratchpad_parameter @p27 : i32
  aiex.scratchpad_parameter @p28 : i32
  aiex.scratchpad_parameter @p29 : i32
  aiex.scratchpad_parameter @p30 : i32
  aiex.scratchpad_parameter @p31 : i32
  aiex.scratchpad_parameter @p32 : i32
  aie.device(npu2) @main {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %v0 = aiex.read_scratchpad_parameter @p0 : i32
      %v1 = aiex.read_scratchpad_parameter @p1 : i32
      %v2 = aiex.read_scratchpad_parameter @p2 : i32
      %v3 = aiex.read_scratchpad_parameter @p3 : i32
      %v4 = aiex.read_scratchpad_parameter @p4 : i32
      %v5 = aiex.read_scratchpad_parameter @p5 : i32
      %v6 = aiex.read_scratchpad_parameter @p6 : i32
      %v7 = aiex.read_scratchpad_parameter @p7 : i32
      %v8 = aiex.read_scratchpad_parameter @p8 : i32
      %v9 = aiex.read_scratchpad_parameter @p9 : i32
      %v10 = aiex.read_scratchpad_parameter @p10 : i32
      %v11 = aiex.read_scratchpad_parameter @p11 : i32
      %v12 = aiex.read_scratchpad_parameter @p12 : i32
      %v13 = aiex.read_scratchpad_parameter @p13 : i32
      %v14 = aiex.read_scratchpad_parameter @p14 : i32
      %v15 = aiex.read_scratchpad_parameter @p15 : i32
      %v16 = aiex.read_scratchpad_parameter @p16 : i32
      %v17 = aiex.read_scratchpad_parameter @p17 : i32
      %v18 = aiex.read_scratchpad_parameter @p18 : i32
      %v19 = aiex.read_scratchpad_parameter @p19 : i32
      %v20 = aiex.read_scratchpad_parameter @p20 : i32
      %v21 = aiex.read_scratchpad_parameter @p21 : i32
      %v22 = aiex.read_scratchpad_parameter @p22 : i32
      %v23 = aiex.read_scratchpad_parameter @p23 : i32
      %v24 = aiex.read_scratchpad_parameter @p24 : i32
      %v25 = aiex.read_scratchpad_parameter @p25 : i32
      %v26 = aiex.read_scratchpad_parameter @p26 : i32
      %v27 = aiex.read_scratchpad_parameter @p27 : i32
      %v28 = aiex.read_scratchpad_parameter @p28 : i32
      %v29 = aiex.read_scratchpad_parameter @p29 : i32
      %v30 = aiex.read_scratchpad_parameter @p30 : i32
      %v31 = aiex.read_scratchpad_parameter @p31 : i32
      %v32 = aiex.read_scratchpad_parameter @p32 : i32
      aie.end
    }
    aie.runtime_sequence @seq() {
    }
  }
}

// -----

// ERR: error: parameter 'b' and another parameter command 'main:seq' reads both have state-table entry 0
module {
  aiex.scratchpad_parameter @a : i32 {state_table_idx = 0 : ui8}
  aiex.scratchpad_parameter @b : i32 {state_table_idx = 0 : ui8}
  aie.device(npu2) @main {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %a = aiex.read_scratchpad_parameter @a : i32
      %b = aiex.read_scratchpad_parameter @b : i32
      aie.end
    }
    aie.runtime_sequence @seq(%arg0 : memref<16xi32>) {
    }
  }
}

// -----

// ERR: error: parameter 'c10' finds no free state-table entry among 32: no assignment gives the parameters each command reads distinct entries
// ERR: note: read by command 'main:bc'
// ERR: note: read by command 'main:ac'
module {
  aiex.scratchpad_parameter @a0 : i32
  aiex.scratchpad_parameter @a1 : i32
  aiex.scratchpad_parameter @a2 : i32
  aiex.scratchpad_parameter @a3 : i32
  aiex.scratchpad_parameter @a4 : i32
  aiex.scratchpad_parameter @a5 : i32
  aiex.scratchpad_parameter @a6 : i32
  aiex.scratchpad_parameter @a7 : i32
  aiex.scratchpad_parameter @a8 : i32
  aiex.scratchpad_parameter @a9 : i32
  aiex.scratchpad_parameter @a10 : i32
  aiex.scratchpad_parameter @b0 : i32
  aiex.scratchpad_parameter @b1 : i32
  aiex.scratchpad_parameter @b2 : i32
  aiex.scratchpad_parameter @b3 : i32
  aiex.scratchpad_parameter @b4 : i32
  aiex.scratchpad_parameter @b5 : i32
  aiex.scratchpad_parameter @b6 : i32
  aiex.scratchpad_parameter @b7 : i32
  aiex.scratchpad_parameter @b8 : i32
  aiex.scratchpad_parameter @b9 : i32
  aiex.scratchpad_parameter @b10 : i32
  aiex.scratchpad_parameter @c0 : i32
  aiex.scratchpad_parameter @c1 : i32
  aiex.scratchpad_parameter @c2 : i32
  aiex.scratchpad_parameter @c3 : i32
  aiex.scratchpad_parameter @c4 : i32
  aiex.scratchpad_parameter @c5 : i32
  aiex.scratchpad_parameter @c6 : i32
  aiex.scratchpad_parameter @c7 : i32
  aiex.scratchpad_parameter @c8 : i32
  aiex.scratchpad_parameter @c9 : i32
  aiex.scratchpad_parameter @c10 : i32
  aie.device(npu2) @a {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      %a0 = aiex.read_scratchpad_parameter @a0 : i32
      %a1 = aiex.read_scratchpad_parameter @a1 : i32
      %a2 = aiex.read_scratchpad_parameter @a2 : i32
      %a3 = aiex.read_scratchpad_parameter @a3 : i32
      %a4 = aiex.read_scratchpad_parameter @a4 : i32
      %a5 = aiex.read_scratchpad_parameter @a5 : i32
      %a6 = aiex.read_scratchpad_parameter @a6 : i32
      %a7 = aiex.read_scratchpad_parameter @a7 : i32
      %a8 = aiex.read_scratchpad_parameter @a8 : i32
      %a9 = aiex.read_scratchpad_parameter @a9 : i32
      %a10 = aiex.read_scratchpad_parameter @a10 : i32
      aie.end
    }
    aie.runtime_sequence @seq_a(%arg0 : memref<16xi32>) {
    }
  }
  aie.device(npu2) @b {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      %b0 = aiex.read_scratchpad_parameter @b0 : i32
      %b1 = aiex.read_scratchpad_parameter @b1 : i32
      %b2 = aiex.read_scratchpad_parameter @b2 : i32
      %b3 = aiex.read_scratchpad_parameter @b3 : i32
      %b4 = aiex.read_scratchpad_parameter @b4 : i32
      %b5 = aiex.read_scratchpad_parameter @b5 : i32
      %b6 = aiex.read_scratchpad_parameter @b6 : i32
      %b7 = aiex.read_scratchpad_parameter @b7 : i32
      %b8 = aiex.read_scratchpad_parameter @b8 : i32
      %b9 = aiex.read_scratchpad_parameter @b9 : i32
      %b10 = aiex.read_scratchpad_parameter @b10 : i32
      aie.end
    }
    aie.runtime_sequence @seq_b(%arg0 : memref<16xi32>) {
    }
  }
  aie.device(npu2) @c {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      %c0 = aiex.read_scratchpad_parameter @c0 : i32
      %c1 = aiex.read_scratchpad_parameter @c1 : i32
      %c2 = aiex.read_scratchpad_parameter @c2 : i32
      %c3 = aiex.read_scratchpad_parameter @c3 : i32
      %c4 = aiex.read_scratchpad_parameter @c4 : i32
      %c5 = aiex.read_scratchpad_parameter @c5 : i32
      %c6 = aiex.read_scratchpad_parameter @c6 : i32
      %c7 = aiex.read_scratchpad_parameter @c7 : i32
      %c8 = aiex.read_scratchpad_parameter @c8 : i32
      %c9 = aiex.read_scratchpad_parameter @c9 : i32
      %c10 = aiex.read_scratchpad_parameter @c10 : i32
      aie.end
    }
    aie.runtime_sequence @seq_c(%arg0 : memref<16xi32>) {
    }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @ab(%arg0 : memref<16xi32>) {
      aiex.configure @a {
        aiex.run @seq_a(%arg0) : (memref<16xi32>)
      }
      aiex.configure @b {
        aiex.run @seq_b(%arg0) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @bc(%arg0 : memref<16xi32>) {
      aiex.configure @b {
        aiex.run @seq_b(%arg0) : (memref<16xi32>)
      }
      aiex.configure @c {
        aiex.run @seq_c(%arg0) : (memref<16xi32>)
      }
    }
    aie.runtime_sequence @ac(%arg0 : memref<16xi32>) {
      aiex.configure @a {
        aiex.run @seq_a(%arg0) : (memref<16xi32>)
      }
      aiex.configure @c {
        aiex.run @seq_c(%arg0) : (memref<16xi32>)
      }
    }
  }
}

// -----

// ERR: error: parameter 'a' has state-table entry 40, past the 32 there are
module {
  aiex.scratchpad_parameter @a : i32 {state_table_idx = 40 : ui8}
  aie.device(npu2) @main {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %a = aiex.read_scratchpad_parameter @a : i32
      aie.end
    }
    aie.runtime_sequence @seq(%arg0 : memref<16xi32>) {
    }
  }
}
