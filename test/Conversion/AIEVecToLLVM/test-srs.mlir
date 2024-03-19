// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm | FileCheck %s
func.func @srs() {
  %v32i32 = llvm.mlir.undef : vector<32xi32>
  // sweep the variants of SRS and values of shift
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %v32i32, %c0 : vector<32xi32>, i32, vector<32xi16>
  %1 = aievec.srs %v32i32, %c5 : vector<32xi32>, i32, vector<32xi16>
  return
}

// CHECK: [[UNDEF_V32I32:%.+]] = llvm.mlir.undef : vector<32xi32>
// CHECK: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %0 : vector<32xi32> to vector<16xi64>
// CHECK: %[[SRS0:.*]] = "xllvm.intr.aie2.I512.v32.acc32.srs"(
// CHECK-SAME: [[BITCAST0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<32xi16>
// CHECK: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[BITCAST1:.*]] = llvm.bitcast %0 : vector<32xi32> to vector<16xi64>
// CHECK: %[[SRS1:.*]] = "xllvm.intr.aie2.I512.v32.acc32.srs"(
// CHECK-SAME: [[BITCAST1]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<32xi16>
