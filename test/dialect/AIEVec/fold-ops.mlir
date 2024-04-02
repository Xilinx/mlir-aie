// RUN: aie-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_srs_ups_fold(
// CHECK-SAME: %[[INV:.*]]: vector<8xi80>
func.func @test_srs_ups_fold(%in : vector<8xi80>) -> vector<8xi80> {
    %c0 = arith.constant 0 : i32
    %0 = aievec.srs %in, %c0 : vector<8xi80>, i32, vector<8xi32>
    %1 = aievec.ups %0 {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
    // CHECK: return %[[INV]] : vector<8xi80>
    return %1 : vector<8xi80>
}

// -----

// CHECK-LABEL: func.func @test_ups_srs_fold(
// CHECK-SAME: %[[INV:.*]]: vector<16xi16>
func.func @test_ups_srs_fold(%in : vector<16xi16>) -> vector<16xi16> {
    %c0 = arith.constant 0 : i32
    %0 = aievec.ups %in {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
    %1 = aievec.srs %0, %c0 : vector<16xi48>, i32, vector<16xi16>
    // CHECK: return %[[INV]] : vector<16xi16>
    return %1 : vector<16xi16>
}

// -----

// CHECK-LABEL: func.func @test_cast_cast_fold(
// CHECK-SAME: %[[IN:.*]]: vector<16xi16>
func.func @test_cast_cast_fold(%in : vector<16xi16>) -> vector<16xi16> {
    %0 = aievec.cast %in {isResAcc = false} : vector<16xi16>, vector<16xi16>
    %1 = aievec.cast %0 {isResAcc = true} : vector<16xi16>, vector<16xi16>
    // CHECK: return %[[IN]] : vector<16xi16>
    return %1 : vector<16xi16>
}

// -----

// CHECK-LABEL: func.func @test_cast_cast_fold_2(
// CHECK-SAME: %[[IN:.*]]: vector<16xi16>
func.func @test_cast_cast_fold_2(%in : vector<16xi16>) -> vector<16xi16> {
    %0 = aievec.cast %in {isResAcc = true} : vector<16xi16>, vector<16xi16>
    %1 = aievec.cast %0 {isResAcc = false} : vector<16xi16>, vector<16xi16>
    // CHECK: return %[[IN]] : vector<16xi16>
    return %1 : vector<16xi16>
}
