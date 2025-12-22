//===- error_cyclic.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file --aie-materialize-runtime-sequences %s

// Test detection of self-recursive runtime sequence calls

module {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)
    
    // expected-error @+1 {{Runtime sequence call graph contains a cycle}}
    aie.runtime_sequence @main_seq(%arg0: memref<16xi32>) {
      aiex.configure @main {
        aiex.run @main_seq(%arg0) : (memref<16xi32>)
      }
    }
  }
}

// -----

// Test detection of cycles through intermediate runtime sequences

module {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)
    
    // expected-error @+1 {{Runtime sequence call graph contains a cycle}}
    aie.runtime_sequence @main_seq(%arg0: memref<16xi32>) {
      aiex.configure @config_a {
        aiex.run @seq_a(%arg0) : (memref<16xi32>)
      }
    }
  }
  
  aie.device(npu2) @config_a {
    %tile10 = aie.tile(1, 0)
    
    aie.runtime_sequence @seq_a(%arg0: memref<16xi32>) {
      aiex.configure @config_b {
        aiex.run @seq_b(%arg0) : (memref<16xi32>)
      }
    }
  }
  
  aie.device(npu2) @config_b {
    %tile20 = aie.tile(2, 0)
    
    aie.runtime_sequence @seq_b(%arg0: memref<16xi32>) {
      // This creates a cycle: main -> a -> b -> a
      aiex.configure @config_a {
        aiex.run @seq_a(%arg0) : (memref<16xi32>)
      }
    }
  }
}
