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

// -----

// CHECK-LABEL: @mul_i16
// CHECK-SAME: %[[A:.*]]: vector<16xi16>,
// CHECK-SAME: %[[B:.*]]: vector<16xi16>
// CHECK:      %[[RES:.*]] = aievec_aie1.mul %[[A]], %[[B]] :
// CHECK-SAME: vector<16xi16>, vector<16xi16>, vector<16xi48>
// CHECK: return %[[RES]] : vector<16xi48>
func.func @mul_i16(%A : vector<16xi16>, %B : vector<16xi16>) -> vector<16xi48> {
  %0 = aievec_aie1.mul %A, %B : vector<16xi16>, vector<16xi16>, vector<16xi48>
  return %0 : vector<16xi48>
}

// -----

// CHECK-LABEL: @mac_f32
// CHECK-SAME: %[[A:.*]]: vector<16xf32>,
// CHECK-SAME: %[[B:.*]]: vector<8xf32>,
// CHECK-SAME: %[[ACC:.*]]: vector<8xf32>
// CHECK:      %[[RES:.*]] = aievec_aie1.mac %[[A]], %[[B]], %[[ACC]] 
// CHECK-SAME: {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : 
// CHECK-SAME: vector<16xf32>, vector<8xf32>, vector<8xf32>
// CHECK: return %[[RES]] : vector<8xf32>
func.func @mac_f32(%A : vector<16xf32>, %B : vector<8xf32>, %ACC : vector<8xf32>) -> vector<8xf32> {
  %0 = aievec_aie1.mac %A, %B, %ACC {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}
