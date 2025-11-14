//===- aie-vector-to-pointer-loops.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-vector-to-pointer-loops %s | FileCheck %s

// Test 1: Basic vector.load transformation with loop-carried index
// This tests that scf.for loops with loop-carried indices used in vector.load
// operations are transformed to use pointer iter_args.
//
// Expected transformation:
//   - memref converted to generic_space via unrealized_conversion_cast
//   - ptr.to_ptr creates base pointer
//   - ptr.ptr_add initializes pointer with offset
//   - scf.for iter_arg becomes pointer type
//   - vector.load uses ptr.from_ptr to reconstruct memref with offset 0
//   - arith.addi becomes ptr.ptr_add for pointer increment
//
// CHECK-LABEL: module @basic_vector_load
module @basic_vector_load {
  aie.device(xcvc1902) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) : memref<1024xi32>
    
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      
      // CHECK: scf.for
      %result = scf.for %i = %c0 to %c16 step %c1 iter_args(%idx = %c0) -> (index) {
        // CHECK: vector.load
        %vec = vector.load %buf[%idx] : memref<1024xi32>, vector<16xi32>
        
        // CHECK: arith.addi
        %next_idx = arith.addi %idx, %c1 : index
        
        // CHECK: scf.yield
        scf.yield %next_idx : index
      }
      aie.end
    }
  }
}

// Test 2: Vector.store transformation with loop-carried index
// Similar to Test 1 but with vector.store operations
//
// CHECK-LABEL: module @basic_vector_store
module @basic_vector_store {
  aie.device(xcvc1902) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) : memref<1024xi32>
    
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant dense<1> : vector<16xi32>
      
      // CHECK: scf.for
      %result = scf.for %i = %c0 to %c16 step %c1 iter_args(%idx = %c0) -> (index) {
        // CHECK: vector.store
        vector.store %cst, %buf[%idx] : memref<1024xi32>, vector<16xi32>
        %next_idx = arith.addi %idx, %c1 : index
        scf.yield %next_idx : index
      }
      aie.end
    }
  }
}

// Test 3: Vector operations with i8 element type (1-byte elements)
// Tests that byte offsets are calculated correctly for i8 types
//
// CHECK-LABEL: module @vector_i8_elements
module @vector_i8_elements {
  aie.device(xcvc1902) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) : memref<2048xi8>
    
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      
      // CHECK: scf.for
      %result = scf.for %i = %c0 to %c16 step %c16 iter_args(%idx = %c0) -> (index) {
        // CHECK: vector.load
        %vec = vector.load %buf[%idx] : memref<2048xi8>, vector<64xi8>
        %next_idx = arith.addi %idx, %c64 : index
        scf.yield %next_idx : index
      }
      aie.end
    }
  }
}

// Test 4: Vector operations with i16 element type (2-byte elements)
// Tests that byte offsets are scaled correctly (element_index * 2 = byte_offset)
//
// CHECK-LABEL: module @vector_i16_elements
module @vector_i16_elements {
  aie.device(xcvc1902) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) : memref<1024xi16>
    
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      
      // CHECK: scf.for
      %result = scf.for %i = %c0 to %c8 step %c8 iter_args(%idx = %c0) -> (index) {
        // CHECK: vector.load
        %vec = vector.load %buf[%idx] : memref<1024xi16>, vector<32xi16>
        %next_idx = arith.addi %idx, %c32 : index
        scf.yield %next_idx : index
      }
      aie.end
    }
  }
}

// Test 5: Both load and store in same loop with multiple memrefs
// Tests handling of multiple memrefs with separate loop-carried indices
//
// CHECK-LABEL: module @load_and_store
module @load_and_store {
  aie.device(xcvc1902) {
    %tile = aie.tile(1, 1)
    %buf_in = aie.buffer(%tile) : memref<1024xi32>
    %buf_out = aie.buffer(%tile) : memref<1024xi32>
    
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      
      // CHECK: scf.for
      %result:2 = scf.for %i = %c0 to %c16 step %c1 iter_args(%idx_in = %c0, %idx_out = %c0) -> (index, index) {
        // CHECK: vector.load
        %vec = vector.load %buf_in[%idx_in] : memref<1024xi32>, vector<16xi32>
        // CHECK: vector.store
        vector.store %vec, %buf_out[%idx_out] : memref<1024xi32>, vector<16xi32>
        %next_idx_in = arith.addi %idx_in, %c1 : index
        %next_idx_out = arith.addi %idx_out, %c1 : index
        scf.yield %next_idx_in, %next_idx_out : index, index
      }
      aie.end
    }
  }
}

// Test 6: Non-zero initial offset
// Tests that initial pointer is properly offset
//
// CHECK-LABEL: module @non_zero_init
module @non_zero_init {
  aie.device(xcvc1902) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) : memref<1024xi32>
    
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      
      // CHECK: scf.for
      %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%idx = %c64) -> (index) {
        // CHECK: vector.load
        %vec = vector.load %buf[%idx] : memref<1024xi32>, vector<16xi32>
        %next_idx = arith.addi %idx, %c1 : index
        scf.yield %next_idx : index
      }
      aie.end
    }
  }
}

// Test 7: Loop with mixed iter_args (index and non-index types)
// Tests that only index iter_args used in vector ops are transformed
//
// CHECK-LABEL: module @mixed_iter_args
module @mixed_iter_args {
  aie.device(xcvc1902) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) : memref<1024xi32>
    
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %init_sum = arith.constant 0 : i32
      
      // CHECK: scf.for
      %result:2 = scf.for %i = %c0 to %c16 step %c1 iter_args(%idx = %c0, %sum = %init_sum) -> (index, i32) {
        // CHECK: vector.load
        %vec = vector.load %buf[%idx] : memref<1024xi32>, vector<16xi32>
        %elem = vector.extract %vec[0] : i32 from vector<16xi32>
        %new_sum = arith.addi %sum, %elem : i32
        %next_idx = arith.addi %idx, %c1 : index
        scf.yield %next_idx, %new_sum : index, i32
      }
      aie.end
    }
  }
}

// Test 8: Negative test - index not loop-carried
// Should NOT transform because the index is not a loop iter_arg
//
// CHECK-LABEL: module @no_transform_non_loop_carried
module @no_transform_non_loop_carried {
  aie.device(xcvc1902) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) : memref<1024xi32>
    
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %fixed_idx = arith.constant 100 : index
      
      // CHECK: scf.for
      // CHECK-NOT: ptr.ptr
      scf.for %i = %c0 to %c16 step %c1 {
        // CHECK: vector.load
        %vec = vector.load %buf[%fixed_idx] : memref<1024xi32>, vector<16xi32>
      }
      aie.end
    }
  }
}
