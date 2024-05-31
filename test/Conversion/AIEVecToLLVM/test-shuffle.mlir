// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm -canonicalize | FileCheck %s

// CHECK-LABEL: @shuffle_single_operand_nocast
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>
func.func @shuffle_single_operand_nocast(%lhs : vector<16xi32>)
                -> vector<16xi32> {
  // CHECK: %[[M:.*]] = llvm.mlir.constant(34 : i32) : i32
  // CHECK: %[[RHS:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
  // CHECK: %[[R:.*]] = "xllvm.intr.aie2.vshuffle"(%[[LHS]], %[[RHS]], %[[M]]) :
  // CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  %0 = aievec.shuffle %lhs [t32_4x4] : vector<16xi32>
  // CHECK: return %[[R]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @shuffle_two_operands_nocast
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>
func.func @shuffle_two_operands_nocast(%lhs : vector<16xi32>,
                                       %rhs : vector<16xi32>)
                -> vector<16xi32> {
  // CHECK: %[[M:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[R:.*]] = "xllvm.intr.aie2.vshuffle"(%[[LHS]], %[[RHS]], %[[M]]) :
  // CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  %0 = aievec.shuffle %lhs, %rhs [t32_4x8_lo] : vector<16xi32>
  // CHECK: return %[[R]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @shuffle_single_operand_cast
// CHECK-SAME: %[[V:.*]]: vector<32xbf16>
func.func @shuffle_single_operand_cast(%lhs : vector<32xbf16>)
                -> vector<32xbf16> {
  // CHECK: %[[M:.*]] = llvm.mlir.constant(42 : i32) : i32
  // CHECK: %[[RHS:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
  // CHECK: %[[LHS:.]] = llvm.bitcast %[[V]] : vector<32xbf16> to vector<16xi32>
  // CHECK: %[[S:.*]] = "xllvm.intr.aie2.vshuffle"(%[[LHS]], %[[RHS]], %[[M]]) :
  // CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  // CHECK: %[[R:.]] = llvm.bitcast %[[S]] : vector<16xi32> to vector<32xbf16>
  %0 = aievec.shuffle %lhs [t16_8x2] : vector<32xbf16>
  // CHECK: return %[[R]] : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// CHECK-LABEL: @shuffle_two_operands_cast
// CHECK-SAME: %[[LV:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[RV:.*]]: vector<32xbf16>
func.func @shuffle_two_operands_cast(%lhs : vector<32xbf16>,
                                     %rhs : vector<32xbf16>)
                -> vector<32xbf16> {
  // CHECK: %[[M:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK: %[[L:.*]] = llvm.bitcast %[[LV]] : vector<32xbf16> to vector<16xi32>
  // CHECK: %[[R:.*]] = llvm.bitcast %[[RV]] : vector<32xbf16> to vector<16xi32>
  // CHECK: %[[S:.*]] = "xllvm.intr.aie2.vshuffle"(%[[L]], %[[R]], %[[M]]) :
  // CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  // CHECK: %[[R:.]] = llvm.bitcast %[[S]] : vector<16xi32> to vector<32xbf16>
  %0 = aievec.shuffle %lhs, %rhs [t16_16x4_lo] : vector<32xbf16>
  // CHECK: return %[[R]] : vector<32xbf16>
  return %0 : vector<32xbf16>
}
