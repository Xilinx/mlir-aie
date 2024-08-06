// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s

// CHECK-LABEL:func @vecmax_i32
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecmax_i32(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[RES:.*]] = aievec.max %[[LHS]], %[[RHS]] : vector<16xi32>
  %0 = arith.maxsi %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL:func @vecmax_i16
// CHECK-SAME: %[[LHS:.*]]: vector<32xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi16>)
func.func @vecmax_i16(%arg0: vector<32xi16>, %arg1: vector<32xi16>) -> vector<32xi16> {
  // CHECK: %[[RES:.*]] = aievec.max %[[LHS]], %[[RHS]] : vector<32xi16>
  %0 = arith.maxsi %arg0, %arg1 : vector<32xi16>
  // CHECK: return %[[RES]] : vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL:func @vecmax_i8
// CHECK-SAME: %[[LHS:.*]]: vector<64xi8>,
// CHECK-SAME: %[[RHS:.*]]: vector<64xi8>)
func.func @vecmax_i8(%arg0: vector<64xi8>, %arg1: vector<64xi8>) -> vector<64xi8> {
  // CHECK: %[[RES:.*]] = aievec.max %[[LHS]], %[[RHS]] : vector<64xi8>
  %0 = arith.maxsi %arg0, %arg1 : vector<64xi8>
  // CHECK: return %[[RES]] : vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL:func @vecmax_bf16
// CHECK-SAME: %[[LHS:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xbf16>)
func.func @vecmax_bf16(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: %[[RES:.*]] = aievec.max %[[LHS]], %[[RHS]] : vector<32xbf16>
  %0 = arith.maximumf %arg0, %arg1 : vector<32xbf16>
  // CHECK: return %[[RES]] : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL:func @vecmax_f32
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecmax_f32(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[RES:.*]] = aievec.max %[[LHS]], %[[RHS]] : vector<16xf32>
  %0 = arith.maximumf %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xf32>
  return %0 : vector<16xf32>
}

