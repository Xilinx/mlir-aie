// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s

// CHECK-LABEL: func @test_mac_elem
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<16xi32>
func.func @test_mac_elem(%a : vector<16xi32>,
                         %b : vector<16xi32>,
                         %c : vector<16xi32>) -> vector<16xi32> {
    // CHECK:  %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[UPS:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
    // CHECK: %[[ME:.*]] = aievec.mac_elem %[[A]], %[[B]], %[[UPS:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    // CHECK: %[[RES:.*]] = aievec.srs %[[ME:.*]], %[[C0]] : vector<16xi64>, i32, vector<16xi32>
    %0 = arith.muli %a, %b : vector<16xi32>
    %1 = arith.addi %0, %c : vector<16xi32>
    // CHECK: return %[[RES:.*]] : vector<16xi32>
    return %1 : vector<16xi32>
}

// CHECK-LABEL: func @test_mac_elem_inv
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<16xi32>
func.func @test_mac_elem_inv(%a : vector<16xi32>,
                             %b : vector<16xi32>,
                             %c : vector<16xi32>) -> vector<16xi32> {
    // CHECK:  %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[UPS:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
    // CHECK: %[[ME:.*]] = aievec.mac_elem %[[A]], %[[B]], %[[UPS:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
    // CHECK: %[[RES:.*]] = aievec.srs %[[ME:.*]], %[[C0]] : vector<16xi64>, i32, vector<16xi32>
    %0 = arith.muli %a, %b : vector<16xi32>
    %1 = arith.addi %c, %0 : vector<16xi32>
    // CHECK: return %[[RES:.*]] : vector<16xi32>
    return %1 : vector<16xi32>
}
