// RUN: aie-opt %s -canonicalize-for-aievec -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @splat(
// CHECK-SAME: %[[MEM:.*]]: memref<?xi32>,
// CHECK-SAME: %[[POS:.*]]: index
func.func @splat(%m : memref<?xi32>, %pos : index) -> vector<8xi32> {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    %c0_i32 = arith.constant 0 : i32
    %i = affine.apply affine_map<(d0) -> (d0 + 5)>(%pos)
    // CHECK: %[[V:.*]] = vector.transfer_read %[[MEM]][%[[POS]]], %[[C0]] : memref<?xi32>, vector<8xi32>
    // CHECK: %[[E:.*]] = vector.extract %[[V]][5] : vector<8xi32>
    // CHECK: %[[S:.*]] = vector.broadcast %[[E]] : i32 to vector<8xi32>
    %v = vector.transfer_read %m[%i], %c0_i32 {permutation_map = affine_map<(d0) -> (0)>} : memref<?xi32>, vector<8xi32>
    // CHECK: return %[[S]] : vector<8xi32>
    return %v : vector<8xi32>
}

// CHECK-LABEL: func.func @muladd2fma_i32(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xi32>
func.func @muladd2fma_i32(%a : vector<8xi32>, %b : vector<8xi32>, %c : vector<8xi32>) -> vector<8xi32> {
    // CHECK: %[[R:.*]] = vector.fma %[[A]], %[[B]], %[[C]] : vector<8xi32>
    %0 = arith.muli %a, %b : vector<8xi32>
    %1 = arith.addi %0, %c : vector<8xi32>
    // CHECK: return %[[R]] : vector<8xi32>
    return %1 : vector<8xi32>
}

// CHECK-LABEL: func.func @muladd2fma_f32(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xf32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xf32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xf32>
func.func @muladd2fma_f32(%a : vector<8xf32>, %b : vector<8xf32>, %c : vector<8xf32>) -> vector<8xf32> {
    // CHECK: %[[R:.*]] = vector.fma %[[A]], %[[B]], %[[C]] : vector<8xf32>
    %0 = arith.mulf %a, %b : vector<8xf32>
    %1 = arith.addf %0, %c : vector<8xf32>
    // CHECK: return %[[R]] : vector<8xf32>
    return %1 : vector<8xf32>
}

// CHECK-LABEL: func.func @muladd2fma_inv(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xi32>
func.func @muladd2fma_inv(%a : vector<8xi32>, %b : vector<8xi32>, %c : vector<8xi32>) -> vector<8xi32> {
    // CHECK: %[[R:.*]] = vector.fma %[[A]], %[[B]], %[[C]] : vector<8xi32>
    %0 = arith.muli %a, %b : vector<8xi32>
    %1 = arith.addi %c, %0 : vector<8xi32>
    // CHECK: return %[[R]] : vector<8xi32>
    return %1 : vector<8xi32>
}
