//===- run_error_arg_type_mismatch.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// Test detection of argument type mismatches when calling runtime sequences

module {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)
    
    aie.runtime_sequence @main_seq(%arg0: memref<16xi32>) {
      aiex.configure @config_wrong_type {
        // expected-error @+1 {{argument 0 type mismatch}}
        aiex.run @seq_expects_i16(%arg0) : (memref<16xi32>)
      }
    }
  }
  
  aie.device(npu2) @config_wrong_type {
    %tile10 = aie.tile(1, 0)
    
    // This expects memref<16xi16> but we're passing memref<16xi32>
    aie.runtime_sequence @seq_expects_i16(%arg0: memref<16xi16>) {
      aiex.npu.write32 {address = 100 : ui32, column = 1 : i32, row = 0 : i32, value = 42 : ui32}
    }
  }
}
