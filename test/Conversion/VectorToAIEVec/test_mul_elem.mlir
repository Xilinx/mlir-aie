// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s

// CHECK-LABEL: func @test_mul_elem_i32
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xi32>
func.func @test_mul_elem_i32(%a : vector<16xi32>,
                         %b : vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[ME:.*]] = aievec.mul_elem %[[A]], %[[B]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
  // CHECK: %[[RES:.*]] = aievec.srs %[[ME:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
  %1 = arith.muli %a, %b : vector<16xi32>
  return %1 : vector<16xi32>
}

// CHECK-LABEL: func @test_mul_elem_i16
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xi16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<32xi16>
func.func @test_mul_elem_i16(%a : vector<32xi16>,
                         %b : vector<32xi16>) -> vector<32xi16> {
  // CHECK: %[[ME:.*]] = aievec.mul_elem %[[A]], %[[B]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
  // CHECK: %[[RES:.*]] = aievec.srs %[[ME:.*]] {shift = 0 : i8} : vector<32xi32>, vector<32xi16>
  %1 = arith.muli %a, %b : vector<32xi16>
  return %1 : vector<32xi16>
}

func.func @test_mul_elem_i8(%a : vector<32xi8>,
                         %b : vector<32xi8>) -> vector<32xi32> {
  %1 = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %2 = arith.extsi %b : vector<32xi8> to vector<32xi32>
  // CHECK:  %[[BCS:.*]] = aievec.broadcast_scalar {scalar = 0 : i32} : vector<64xi8>
  // CHECK:  %[[EXT:.*]] = aievec.ext %[[BCS:.*]] {index = 0 : i8} : vector<64xi8>, vector<32xi8>
  // CHECK:  %[[CC1:.*]] = aievec.concat %[[A]], %[[EXT:.*]] : vector<32xi8>, vector<64xi8>
  // CHECK:  %[[CC2:.*]] = aievec.concat %[[B]], %[[EXT:.*]] : vector<32xi8>, vector<64xi8>
  // CHECK:  %[[ME:.*]] = aievec.mul_elem %[[CC1:.*]], %[[CC2:.*]] : vector<64xi8>, vector<64xi8>, vector<32xi32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[ME:.*]] {isResAcc = false} : vector<32xi32>, vector<32xi32>

  %3 = arith.muli %1, %2 : vector<32xi32>
  return %3 : vector<32xi32>
}
