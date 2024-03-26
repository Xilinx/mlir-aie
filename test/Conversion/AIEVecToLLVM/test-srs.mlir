// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm | FileCheck %s
func.func @srs(%arg0 : vector<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<32xi32>, i32, vector<32xi16>
  %1 = aievec.srs %arg0, %c5 : vector<32xi32>, i32, vector<32xi16>
  return
}

// CHECK-LABEL: @srs
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi32>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi32> to vector<16xi64>
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.I512.v32.acc32.srs"(
// CHECK-SAME: [[BITCAST0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<32xi16>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi32> to vector<16xi64>
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.I512.v32.acc32.srs"(
// CHECK-SAME: [[BITCAST1]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<32xi16>
