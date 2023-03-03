// RUN: aie-opt %s --convert-vector-to-aievec | FileCheck %s

// CHECK-LABEL: func.func @muladd2mac_i32(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xi32>
func.func @muladd2mac_i32(%a : vector<8xi32>,
                          %b : vector<8xi32>,
                          %c : vector<8xi32>) -> vector<8xi32> {
    // CHECK: %[[ACC:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    // CHECK: %[[MAC:.*]] = aievec.mac %[[A]], %[[B]], %[[ACC]] : vector<8xi32>, vector<8xi32>, vector<8xi80>
    // CHECK: %[[RES:.*]] = aievec.srs %[[MAC]] {shift = 0 : i8} : vector<8xi80>, vector<8xi32>
    %0 = arith.muli %a, %b : vector<8xi32>
    %1 = arith.addi %0, %c : vector<8xi32>
    // CHECK: return %[[RES]] : vector<8xi32>
    return %1 : vector<8xi32>
}

// CHECK-LABEL: func.func @muladd2mac_inv(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xi32>
func.func @muladd2mac_inv(%a : vector<8xi32>,
                          %b : vector<8xi32>,
                          %c : vector<8xi32>) -> vector<8xi32> {
    // CHECK: %[[ACC:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    // CHECK: %[[MAC:.*]] = aievec.mac %[[A]], %[[B]], %[[ACC]] : vector<8xi32>, vector<8xi32>, vector<8xi80>
    // CHECK: %[[RES:.*]] = aievec.srs %[[MAC]] {shift = 0 : i8} : vector<8xi80>, vector<8xi32>
    %0 = arith.muli %a, %b : vector<8xi32>
    %1 = arith.addi %c, %0 : vector<8xi32>
    // CHECK: return %[[RES]] : vector<8xi32>
    return %1 : vector<8xi32>
}
