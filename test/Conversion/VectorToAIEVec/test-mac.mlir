// RUN: aie-opt %s --convert-vector-to-aievec | FileCheck %s

// CHECK-LABEL: func @test_fma_to_mac(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<8xi32>,
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<8xi32>
func.func @test_fma_to_mac(%a : vector<8xi32>,
                           %b : vector<8xi32>,
                           %c : vector<8xi32>) -> vector<8xi32> {
    // CHECK: %[[ACC:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    // CHECK: %[[MAC:.*]] = aievec.mac %[[A]], %[[B]], %[[ACC]] : vector<8xi32>, vector<8xi32>, vector<8xi80>
    // CHECK: %[[RES:.*]] = aievec.srs %[[MAC]] {shift = 0 : i8} : vector<8xi80>, vector<8xi32>
    %0 = vector.fma %a, %b, %c : vector<8xi32>
    // CHECK: return %[[RES]] : vector<8xi32>
    return %0 : vector<8xi32>
}
