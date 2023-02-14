// RUN: aie-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_srs_ups_fold(
// CHECK-SAME: %[[INV:.*]]: vector<8xi48>
func.func @test_srs_ups_fold(%in : vector<8xi48>) -> vector<8xi48> {
    %0 = aievec.srs %in {shift = 0 : i8} : vector<8xi48>, vector<8xi32>
    %1 = aievec.ups %0 {shift = 0 : i8} : vector<8xi32>, vector<8xi48>
    // CHECK: return %[[INV]] : vector<8xi48>
    return %1 : vector<8xi48>
}
