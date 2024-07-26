// RUN: aie-opt %s --convert-vector-to-aievec | FileCheck %s

// CHECK-LABEL: func @vecaddi(
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecaddi(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[RES:.*]] = aievec_aie1.add %[[LHS]], %[[RHS]] : vector<16xi32>, vector<16xi32>, vector<16xi32>
  %0 = arith.addi %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: func @vecaddf(
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecaddf(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[RES:.*]] = aievec_aie1.add %[[LHS]], %[[RHS]] : vector<16xf32>, vector<16xf32>, vector<16xf32>
  %0 = arith.addf %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @vecsubi(
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecsubi(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[RES:.*]] = aievec_aie1.sub %[[LHS]], %[[RHS]] : vector<16xi32>, vector<16xi32>, vector<16xi32>
  %0 = arith.subi %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: func @vecsubf(
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecsubf(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[RES:.*]] = aievec_aie1.sub %[[LHS]], %[[RHS]] : vector<16xf32>, vector<16xf32>, vector<16xf32>
  %0 = arith.subf %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @vecmuli16(
// CHECK-SAME: %[[LHS:.*]]: vector<16xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi16>)
func.func @vecmuli16(%arg0: vector<16xi16>, %arg1: vector<16xi16>) -> vector<16xi16> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[MUL:.*]] = aievec_aie1.mul %[[LHS]], %[[RHS]] : vector<16xi16>, vector<16xi16>, vector<16xi48>
  // CHECK: %[[RES:.*]] = aievec.srs %[[MUL]], %[[C0]] : vector<16xi48>, i32, vector<16xi16>
  %0 = arith.muli %arg0, %arg1 : vector<16xi16>
  // CHECK: return %[[RES]] : vector<16xi16>
  return %0 : vector<16xi16>
}

// CHECK-LABEL: func @vecmuli32(
// CHECK-SAME: %[[LHS:.*]]: vector<8xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<8xi32>)
func.func @vecmuli32(%arg0: vector<8xi32>, %arg1: vector<8xi32>) -> vector<8xi32> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[MUL:.*]] = aievec_aie1.mul %[[LHS]], %[[RHS]] : vector<8xi32>, vector<8xi32>, vector<8xi80>
  // CHECK: %[[RES:.*]] = aievec.srs %[[MUL]], %[[C0]] : vector<8xi80>, i32, vector<8xi32>
  %0 = arith.muli %arg0, %arg1 : vector<8xi32>
  // CHECK: return %[[RES]] : vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: func @vecmulf(
// CHECK-SAME: %[[LHS:.*]]: vector<8xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<8xf32>)
func.func @vecmulf(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  // CHECK: %[[RES:.*]] = aievec_aie1.mul %[[LHS]], %[[RHS]] : vector<8xf32>, vector<8xf32>, vector<8xf32>
  %0 = arith.mulf %arg0, %arg1 : vector<8xf32>
  // CHECK: return %[[RES]] : vector<8xf32>
  return %0 : vector<8xf32>
}
