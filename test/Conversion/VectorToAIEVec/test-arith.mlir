// RUN: aie-opt %s --convert-vector-to-aievec | FileCheck %s

// CHECK-LABEL: func @vecaddi(
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecaddi(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[RES:.*]] = aievec.add %[[LHS]], %[[RHS]] : vector<16xi32>, vector<16xi32>, vector<16xi32>
  %0 = arith.addi %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: func @vecaddf(
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecaddf(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[RES:.*]] = aievec.add %[[LHS]], %[[RHS]] : vector<16xf32>, vector<16xf32>, vector<16xf32>
  %0 = arith.addf %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @vecsubi(
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecsubi(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[RES:.*]] = aievec.sub %[[LHS]], %[[RHS]] : vector<16xi32>, vector<16xi32>, vector<16xi32>
  %0 = arith.subi %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: func @vecsubf(
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecsubf(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[RES:.*]] = aievec.sub %[[LHS]], %[[RHS]] : vector<16xf32>, vector<16xf32>, vector<16xf32>
  %0 = arith.subf %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xf32>
  return %0 : vector<16xf32>
}
