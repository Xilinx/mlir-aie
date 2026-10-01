//===- aie-hoist-vector-transfer-pointers.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-hoist-vector-transfer-pointers -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @hoist_vector_transfer_read
func.func @hoist_vector_transfer_read(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR0:.*]] = %{{.*}}, %[[PTR1:.*]] = %{{.*}})
  scf.for %i = %c0 to %c64 step %c1 {
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR0]]]{{.*}}{in_bounds = [true]}
    %v = vector.transfer_read %arg0[%i], %cst {in_bounds = [true]} : memref<256xf32>, vector<16xf32>
    // CHECK: arith.addi %[[PTR0]]
    vector.transfer_write %v, %arg1[%i] {in_bounds = [true]} : vector<16xf32>, memref<256xf32>
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR1]]]{{.*}}{in_bounds = [true]}
    // CHECK: arith.addi %[[PTR1]]
    // CHECK: scf.yield %{{.*}}, %{{.*}}
  }
  return
}

// -----

// CHECK-LABEL: func.func @hoist_vector_transfer_write
func.func @hoist_vector_transfer_write(%arg0: memref<256xf32>, %arg1: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR:.*]] = %{{.*}})
  scf.for %i = %c0 to %c64 step %c1 {
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR]]] {in_bounds = [true]}
    vector.transfer_write %arg1, %arg0[%i] {in_bounds = [true]} : vector<16xf32>, memref<256xf32>
    // CHECK: arith.addi %[[PTR]], %{{.*}}
    // CHECK: scf.yield %{{.*}}
  }
  return
}

// -----

// CHECK-LABEL: func.func @hoist_2d_memref
func.func @hoist_2d_memref(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32
  
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR0:.*]] = %{{.*}}, %[[PTR1:.*]] = %{{.*}})
  scf.for %i = %c0 to %c16 step %c1 {
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR0]]]{{.*}}{in_bounds = [true]}
    %v = vector.transfer_read %arg0[%i, %c0], %cst {in_bounds = [true]} : memref<16x16xf32>, vector<16xf32>
    // CHECK: arith.addi %[[PTR0]], %{{.*}}
    vector.transfer_write %v, %arg1[%i, %c0] {in_bounds = [true]} : vector<16xf32>, memref<16x16xf32>
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR1]]]{{.*}}{in_bounds = [true]}
    // CHECK: arith.addi %[[PTR1]], %{{.*}}
    // CHECK: scf.yield %{{.*}}, %{{.*}}
  }
  return
}

// -----

// Test strided memref from subview - should preserve stride information in collapse_shape
// CHECK-LABEL: func.func @hoist_strided_memref
func.func @hoist_strided_memref(%arg0: memref<16x16x4x4xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant dense<0.0> : vector<1x1x4x4xf32>
  
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  scf.for %i = %c0 to %c16 step %c1 {
    // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    scf.for %j = %c0 to %c16 step %c1 {
      // CHECK: memref.subview %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1]
      // CHECK-SAME: memref<16x16x4x4xf32, 2> to memref<1x1x4x4xf32, strided<[256, 16, 4, 1], offset: ?>, 2>
      %subview = memref.subview %arg0[%i, %j, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] 
        : memref<16x16x4x4xf32, 2> to memref<1x1x4x4xf32, strided<[256, 16, 4, 1], offset: ?>, 2>
      // Subviews are created inside nested loops, so the pass should skip transforming them
      // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
      vector.transfer_write %cst, %subview[%c0, %c0, %c0, %c0] 
        {in_bounds = [true, true, true, true]} : vector<1x1x4x4xf32>, memref<1x1x4x4xf32, strided<[256, 16, 4, 1], offset: ?>, 2>
    }
  }
  return
}

// -----

// Test collapse_shape preserves contiguous strided layout (offset:0 gets canonicalized)
// CHECK-LABEL: func.func @preserve_contiguous_strided_layout
func.func @preserve_contiguous_strided_layout(%arg0: memref<16x16xf32, 2>, %arg1: memref<16x16xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32
  
  // Create contiguous strided subviews outside loop
  %subview0 = memref.subview %arg0[0, 0] [16, 16] [1, 1] 
    : memref<16x16xf32, 2> to memref<16x16xf32, strided<[16, 1], offset: 0>, 2>
  %subview1 = memref.subview %arg1[0, 0] [16, 16] [1, 1] 
    : memref<16x16xf32, 2> to memref<16x16xf32, strided<[16, 1], offset: 0>, 2>
  
  // Offset of 0 gets canonicalized away in the output
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK-SAME: memref<16x16xf32, strided<[16, 1]>, 2> into memref<256xf32, strided<[1]>, 2>
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK-SAME: memref<16x16xf32, strided<[16, 1]>, 2> into memref<256xf32, strided<[1]>, 2>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR0:.*]] = %{{.*}}, %[[PTR1:.*]] = %{{.*}})
  scf.for %i = %c0 to %c16 step %c1 {
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR0]]]{{.*}}{in_bounds = [true]}
    %v = vector.transfer_read %subview0[%i, %c0], %cst {in_bounds = [true]}
      : memref<16x16xf32, strided<[16, 1], offset: 0>, 2>, vector<16xf32>
    // CHECK: arith.addi %[[PTR0]], %{{.*}}
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR1]]]{{.*}}{in_bounds = [true]}
    vector.transfer_write %v, %subview1[%i, %c0] {in_bounds = [true]}
      : vector<16xf32>, memref<16x16xf32, strided<[16, 1], offset: 0>, 2>
    // CHECK: arith.addi %[[PTR1]], %{{.*}}
    // CHECK: scf.yield %{{.*}}, %{{.*}}
  }
  return
}

// -----

// Test that loops with only IV-dependent subviews don't cause infinite loops
// This pattern previously caused the pass to hang due to:
// 1. Exponential recursion in dependency checking without memoization
// 2. Infinite loop in greedy rewriter (pattern matches but can't transform)
// The fix adds memoization and early exit for unprocessable transfers
// CHECK-LABEL: func.func @loop_with_iv_dependent_subviews
func.func @loop_with_iv_dependent_subviews(%arg0: memref<1x256xbf16, 2>, %arg1: memref<1x256xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %poison_bf16 = ub.poison : bf16
  %poison_f32 = ub.poison : f32
  
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  scf.for %i = %c0 to %c256 step %c16 {
    // Subviews are created inside the loop using the IV - can't be hoisted
    // Pattern should recognize these can't be processed and skip without hanging
    // CHECK: memref.subview %{{.*}}[0, %{{.*}}] [1, 16] [1, 1]
    %subview_in = memref.subview %arg0[0, %i] [1, 16] [1, 1] 
      : memref<1x256xbf16, 2> to memref<1x16xbf16, strided<[256, 1], offset: ?>, 2>
    // CHECK: memref.subview %{{.*}}[0, %{{.*}}] [1, 16] [1, 1]
    %subview_out = memref.subview %arg1[0, %i] [1, 16] [1, 1] 
      : memref<1x256xf32, 2> to memref<1x16xf32, strided<[256, 1], offset: ?>, 2>
    
    // Transfers on IV-dependent subviews - should not be transformed or cause hang
    // CHECK: vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}]
    %vec_bf16 = vector.transfer_read %subview_in[%c0, %c0], %poison_bf16 
      {in_bounds = [true]} : memref<1x16xbf16, strided<[256, 1], offset: ?>, 2>, vector<16xbf16>
    %vec_f32 = arith.extf %vec_bf16 : vector<16xbf16> to vector<16xf32>
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
    vector.transfer_write %vec_f32, %subview_out[%c0, %c0] 
      {in_bounds = [true]} : vector<16xf32>, memref<1x16xf32, strided<[256, 1], offset: ?>, 2>
  }
  return
}

// -----

// An offset added inside the loop is part of the first address: the pointer
// starts at lb + 768, not lb.
// CHECK-LABEL: func.func @offset_index
func.func @offset_index(%arg0: memref<1536xbf16>, %arg1: memref<768xbf16>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c768 = arith.constant 768 : index
  %cst = arith.constant 0.0 : bf16
  // CHECK: scf.for %{{.*}} = %c0 to %c768 step %c16 iter_args(%[[PTR:.*]] = %c768, %{{.*}} = %c0)
  scf.for %i = %c0 to %c768 step %c16 {
    %j = arith.addi %i, %c768 : index
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR]]]
    %v = vector.transfer_read %arg0[%j], %cst {in_bounds = [true]} : memref<1536xbf16>, vector<16xbf16>
    vector.transfer_write %v, %arg1[%i] {in_bounds = [true]} : vector<16xbf16>, memref<768xbf16>
  }
  return
}

// -----

// The pointer advances by the index's coefficient times the step: %i * 16
// with step 1 is 16 elements per iteration.
// CHECK-LABEL: func.func @scaled_index
func.func @scaled_index(%arg0: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: scf.for {{.*}} step %c1 iter_args(%[[RD:.*]] = %c0, %[[WR:.*]] = %c0)
  // CHECK: vector.transfer_read %{{.*}}[%[[RD]]]
  // CHECK: arith.addi %[[RD]], %c16
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[WR]]]
  // CHECK: arith.addi %[[WR]], %c16
  scf.for %i = %c0 to %c16 step %c1 {
    %j = arith.muli %i, %c16 : index
    %v = vector.transfer_read %arg0[%j], %cst {in_bounds = [true]} : memref<256xf32>, vector<16xf32>
    %w = arith.addf %v, %v : vector<16xf32>
    vector.transfer_write %w, %arg0[%j] {in_bounds = [true]} : vector<16xf32>, memref<256xf32>
  }
  return
}

// -----

// Not known to be in bounds: flattening it would drop the padding semantics.
// CHECK-LABEL: func.func @not_in_bounds_untouched
func.func @not_in_bounds_untouched(%arg0: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK: vector.transfer_read %{{.*}}[%[[I]]], %{{.*}} : memref<256xf32>, vector<16xf32>
  scf.for %i = %c0 to %c256 step %c1 {
    %v = vector.transfer_read %arg0[%i], %cst : memref<256xf32>, vector<16xf32>
    vector.transfer_write %v, %arg0[%c0] {in_bounds = [true]} : vector<16xf32>, memref<256xf32>
  }
  return
}

// -----

// An index carried by an iter_arg is not a function of the IV.
// CHECK-LABEL: func.func @iter_arg_index_untouched
func.func @iter_arg_index_untouched(%arg0: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: scf.for {{.*}} iter_args(%[[IDX:.*]] = %{{.*}}) -> (index) {
  // CHECK: vector.transfer_read %{{.*}}[%[[IDX]]]
  %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%idx = %c0) -> (index) {
    %v = vector.transfer_read %arg0[%idx], %cst {in_bounds = [true]} : memref<256xf32>, vector<16xf32>
    vector.transfer_write %v, %arg0[%c0] {in_bounds = [true]} : vector<16xf32>, memref<256xf32>
    %n = arith.addi %idx, %c16 : index
    scf.yield %n : index
  }
  return
}

// -----

// A vector<4x16> taken from a 64-wide row is four strided runs, not one
// contiguous one, so it cannot be read through a flat pointer.
// CHECK-LABEL: func.func @non_contiguous_vector_untouched
func.func @non_contiguous_vector_untouched(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  // CHECK-NOT: memref.collapse_shape
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK: vector.transfer_read %{{.*}}[%[[I]], %{{.*}}], %{{.*}} {in_bounds = [true, true]} : memref<64x64xf32>, vector<4x16xf32>
  scf.for %i = %c0 to %c64 step %c4 {
    %v = vector.transfer_read %arg0[%i, %c0], %cst {in_bounds = [true, true]} : memref<64x64xf32>, vector<4x16xf32>
    vector.transfer_write %v, %arg1[%i, %c0] {in_bounds = [true, true]} : vector<4x16xf32>, memref<64x64xf32>
  }
  return
}

// -----

// An index loaded from memory inside the loop cannot be recomputed before it.
// CHECK-LABEL: func.func @loaded_index_untouched
func.func @loaded_index_untouched(%arg0: memref<256xf32>, %offs: memref<16xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK: %[[O:.*]] = memref.load
  // CHECK: %[[J:.*]] = arith.addi %[[I]], %[[O]]
  // CHECK: vector.transfer_read %{{.*}}[%[[J]]]
  scf.for %i = %c0 to %c16 step %c1 {
    %o = memref.load %offs[%i] : memref<16xindex>
    %j = arith.addi %i, %o : index
    %v = vector.transfer_read %arg0[%j], %cst {in_bounds = [true]} : memref<256xf32>, vector<16xf32>
    vector.transfer_write %v, %arg0[%c0] {in_bounds = [true]} : vector<16xf32>, memref<256xf32>
  }
  return
}

// -----

// Leading unit dims do not break contiguity: vector<1x1x8x8> of a
// memref<4x8x8x8> is one run of 64 elements, advanced by one 8x8 block
// (64 elements) per %j, starting at block [1, 0] (512).
// CHECK-LABEL: func.func @leading_unit_dims_hoisted
func.func @leading_unit_dims_hoisted(%arg0: memref<4x8x8x8xf32>, %arg1: memref<4x8x8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.0 : f32
  // CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
  // CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
  // CHECK: memref.collapse_shape %{{.*}} {{\[\[}}0, 1, 2, 3]] : memref<4x8x8x8xf32> into memref<2048xf32>
  // CHECK: scf.for {{.*}} iter_args(%[[RD:.*]] = %[[C512]], %[[WR:.*]] = %[[C512]])
  // CHECK: vector.transfer_read %{{.*}}[%[[RD]]], %{{.*}} {in_bounds = [true]} : memref<2048xf32>, vector<64xf32>
  // CHECK: arith.addi %[[RD]], %[[C64]]
  scf.for %j = %c0 to %c8 step %c1 {
    %v = vector.transfer_read %arg0[%c1, %j, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<4x8x8x8xf32>, vector<1x1x8x8xf32>
    vector.transfer_write %v, %arg1[%c1, %j, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<4x8x8x8xf32>
  }
  return
}

// -----

// The coefficient and offset of an affine.apply index are both honoured:
// d0 * 4 + 8 with step 2 starts at 8 and advances by 8 per iteration.
// CHECK-LABEL: func.func @affine_apply_scaled_offset
func.func @affine_apply_scaled_offset(%arg0: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.0 : f32
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: scf.for {{.*}} step %c2 iter_args(%[[PTR:.*]] = %[[C8]], %{{.*}} = %c0)
  // CHECK: vector.transfer_read %{{.*}}[%[[PTR]]]
  // CHECK: arith.addi %[[PTR]], %[[C8]]
  scf.for %i = %c0 to %c32 step %c2 {
    %j = affine.apply affine_map<(d0) -> (d0 * 4 + 8)>(%i)
    %v = vector.transfer_read %arg0[%j], %cst {in_bounds = [true]} : memref<256xf32>, vector<8xf32>
    vector.transfer_write %v, %arg0[%i] {in_bounds = [true]} : vector<8xf32>, memref<256xf32>
  }
  return
}

// -----

// A subtracted IV runs the pointer backwards.
// CHECK-LABEL: func.func @subtracted_index
func.func @subtracted_index(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c240 = arith.constant 240 : index
  %cst = arith.constant 0.0 : f32
  // CHECK-DAG: %[[CM16:.*]] = arith.constant -16 : index
  // CHECK: scf.for {{.*}} iter_args(%[[PTR:.*]] = %c240, %{{.*}} = %c0)
  // CHECK: vector.transfer_read %{{.*}}[%[[PTR]]]
  // CHECK: arith.addi %[[PTR]], %[[CM16]]
  scf.for %i = %c0 to %c240 step %c16 {
    %j = arith.subi %c240, %i : index
    %v = vector.transfer_read %arg0[%j], %cst {in_bounds = [true]} : memref<256xf32>, vector<16xf32>
    vector.transfer_write %v, %arg1[%i] {in_bounds = [true]} : vector<16xf32>, memref<256xf32>
  }
  return
}

// -----

// Without a constant step the per-iteration stride is unknown.
// CHECK-LABEL: func.func @non_constant_step_untouched
func.func @non_constant_step_untouched(%arg0: memref<256xf32>, %step: index) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK: vector.transfer_read %{{.*}}[%[[I]]]
  scf.for %i = %c0 to %c128 step %step {
    %v = vector.transfer_read %arg0[%i], %cst {in_bounds = [true]} : memref<256xf32>, vector<16xf32>
    vector.transfer_write %v, %arg0[%c0] {in_bounds = [true]} : vector<16xf32>, memref<256xf32>
  }
  return
}

// -----

// A transposing permutation map reads a column, not a contiguous run.
// CHECK-LABEL: func.func @permuted_transfer_untouched
func.func @permuted_transfer_untouched(%arg0: memref<16x16xf32>, %arg1: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32
  // CHECK-NOT: memref.collapse_shape
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
  // CHECK: vector.transfer_read %{{.*}}[%{{.*}}, %[[I]]], %{{.*}} {in_bounds = [true], permutation_map = #{{.*}}} : memref<16x16xf32>, vector<16xf32>
  scf.for %i = %c0 to %c16 step %c1 {
    %v = vector.transfer_read %arg0[%c0, %i], %cst {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<16x16xf32>, vector<16xf32>
    vector.transfer_write %v, %arg1[%i] {in_bounds = [true]} : vector<16xf32>, memref<256xf32>
  }
  return
}
