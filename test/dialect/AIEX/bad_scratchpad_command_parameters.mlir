//===- bad_scratchpad_command_parameters.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A command's scratchpad parameters each take one of the command processor's
// 32 state-table entries, so one command reading 33 is refused.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-lower-scratchpad-parameters %s 2>&1 | FileCheck %s --check-prefix=ERR

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
