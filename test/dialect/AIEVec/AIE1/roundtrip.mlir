// RUN: aie-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @add_i32
// CHECK-SAME: %[[A:.*]]: vector<8xi32>,
// CHECK-SAME: %[[B:.*]]: vector<8xi32>
// CHECK:      %[[RES:.*]] = aievec_aie1.add %[[A]], %[[B]] :
// CHECK-SAME: vector<8xi32>, vector<8xi32>, vector<8xi32>
// CHECK: return %[[RES]] : vector<8xi32>
func.func @add_i32(%A : vector<8xi32>, %B : vector<8xi32>) -> vector<8xi32> {
  %0 = aievec_aie1.add %A, %B : vector<8xi32>, vector<8xi32>, vector<8xi32>
  return %0 : vector<8xi32>
}

// -----

// CHECK-LABEL: @sub_i32
// CHECK-SAME: %[[A:.*]]: vector<8xi32>,
// CHECK-SAME: %[[B:.*]]: vector<8xi32>
// CHECK:      %[[RES:.*]] = aievec_aie1.sub %[[A]], %[[B]] :
// CHECK-SAME: vector<8xi32>, vector<8xi32>, vector<8xi32>
// CHECK: return %[[RES]] : vector<8xi32>
func.func @sub_i32(%A : vector<8xi32>, %B : vector<8xi32>) -> vector<8xi32> {
  %0 = aievec_aie1.sub %A, %B : vector<8xi32>, vector<8xi32>, vector<8xi32>
  return %0 : vector<8xi32>
}
