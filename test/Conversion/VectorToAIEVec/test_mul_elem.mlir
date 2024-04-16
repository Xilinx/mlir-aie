// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s

// CHECK-LABEL: func @test_mul_elem_i32
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xi32>
func.func @test_mul_elem_i32(%a : vector<16xi32>,
                         %b : vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[ME:.*]] = aievec.mul_elem %[[A]], %[[B]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
  // CHECK: %[[RES:.*]] = aievec.srs %[[ME]], %[[C0]] : vector<16xi64>, i32, vector<16xi32>
  %1 = arith.muli %a, %b : vector<16xi32>
  return %1 : vector<16xi32>
}

// CHECK-LABEL: func @test_mul_elem_i16
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xi16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<32xi16>
func.func @test_mul_elem_i16(%a : vector<32xi16>,
                         %b : vector<32xi16>) -> vector<32xi16> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[ME:.*]] = aievec.mul_elem %[[A]], %[[B]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
  // CHECK: %[[RES:.*]] = aievec.srs %[[ME]], %[[C0]] : vector<32xi32>, i32, vector<32xi16>
  %1 = arith.muli %a, %b : vector<32xi16>
  return %1 : vector<32xi16>
}

// CHECK-LABEL: func @test_mul_elem_i16_i32
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xi16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<32xi16>
func.func @test_mul_elem_i16_i32(%a : vector<32xi16>,
                         %b : vector<32xi16>) -> vector<32xi32> {
  %1 = arith.extsi %a : vector<32xi16> to vector<32xi32>
  %2 = arith.extsi %b : vector<32xi16> to vector<32xi32>
  // CHECK: %[[ME:.*]] = aievec.mul_elem %[[A]], %[[B]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
  // CHECK: %[[CAST:.*]] = aievec.cast %[[ME]] {isResAcc = false} : vector<32xi32>, vector<32xi32>
  %3 = arith.muli %1, %2 : vector<32xi32>
  return %3 : vector<32xi32> 
}

func.func @test_mul_elem_i8_i32(%a : vector<32xi8>,
                         %b : vector<32xi8>) -> vector<32xi32> {
  %1 = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %2 = arith.extsi %b : vector<32xi8> to vector<32xi32>
  // CHECK:  %[[C0:.*]] = arith.constant 0 : i8
  // CHECK:  %[[BCS:.*]] = aievec.broadcast_scalar %[[C0]] : i8, vector<64xi8>
  // CHECK:  %[[EXT:.*]] = aievec.ext %[[BCS:.*]] {index = 0 : i8} : vector<64xi8>, vector<32xi8>
  // CHECK:  %[[CC1:.*]] = aievec.concat %[[A]], %[[EXT]] : vector<32xi8>, vector<64xi8>
  // CHECK:  %[[CC2:.*]] = aievec.concat %[[B]], %[[EXT]] : vector<32xi8>, vector<64xi8>
  // CHECK:  %[[ME:.*]] = aievec.mul_elem %[[CC1]], %[[CC2]] : vector<64xi8>, vector<64xi8>, vector<32xi32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[ME]] {isResAcc = false} : vector<32xi32>, vector<32xi32>

  %3 = arith.muli %1, %2 : vector<32xi32>
  return %3 : vector<32xi32>
}

func.func @test_mul_elem_i8_i8(%a : vector<32xi8>,
                         %b : vector<32xi8>) -> vector<32xi8> {
  // CHECK:  %[[C0I32:.*]] = arith.constant 0 : i32
  // CHECK:  %[[C0:.*]] = arith.constant 0 : i8
  // CHECK:  %[[BCS:.*]] = aievec.broadcast_scalar %[[C0]] : i8, vector<64xi8>
  // CHECK:  %[[EXT:.*]] = aievec.ext %[[BCS:.*]] {index = 0 : i8} : vector<64xi8>, vector<32xi8>
  // CHECK:  %[[CC1:.*]] = aievec.concat %[[A]], %[[EXT]] : vector<32xi8>, vector<64xi8>
  // CHECK:  %[[CC2:.*]] = aievec.concat %[[B]], %[[EXT]] : vector<32xi8>, vector<64xi8>
  // CHECK:  %[[ME:.*]] = aievec.mul_elem %[[CC1]], %[[CC2]] : vector<64xi8>, vector<64xi8>, vector<32xi32>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[ME]], %[[C0I32]] : vector<32xi32>, i32, vector<32xi8>
  %1 = arith.muli %a, %b : vector<32xi8>
  return %1 : vector<32xi8>
}

// CHECK-LABEL: func @test_mul_elem_bf16
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_mul_elem_bf16(%a : vector<16xbf16>,
                         %b : vector<16xbf16>) -> vector<16xbf16> {
  // CHECK:  %[[C0I32:.*]] = arith.constant 0 : i32
  // CHECK:  %[[C0:.*]] = arith.constant 0.000000e+00 : bf16
  // CHECK:  %[[BCS:.*]] = aievec.broadcast_scalar %[[C0]] : bf16, vector<32xbf16>
  // CHECK:  %[[EXT:.*]] = aievec.ext %[[BCS]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  // CHECK:  %[[CC1:.*]] = aievec.concat %[[A]], %[[EXT]] : vector<16xbf16>, vector<32xbf16>
  // CHECK:  %[[CC2:.*]] = aievec.concat %[[B]], %[[EXT]] : vector<16xbf16>, vector<32xbf16>
  // CHECK:  %[[ME:.*]] = aievec.mul_elem %[[CC1]], %[[CC2]] : vector<32xbf16>, vector<32xbf16>, vector<16xf32>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[ME]], %[[C0I32]] : vector<16xf32>, i32, vector<16xbf16>
  %1 = arith.mulf %a, %b : vector<16xbf16>
  return %1 : vector<16xbf16>
}

// CHECK-LABEL: func @test_mul_elem_bf16_float
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_mul_elem_bf16_float(%a : vector<16xbf16>,
                         %b : vector<16xbf16>) -> vector<16xf32> {
  // CHECK:  %[[C0:.*]] = arith.constant 0.000000e+00 : bf16
  // CHECK:  %[[BCS:.*]] = aievec.broadcast_scalar %[[C0]] : bf16, vector<32xbf16>
  // CHECK:  %[[EXT:.*]] = aievec.ext %[[BCS]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  // CHECK:  %[[CC1:.*]] = aievec.concat %[[A]], %[[EXT]] : vector<16xbf16>, vector<32xbf16>
  // CHECK:  %[[CC2:.*]] = aievec.concat %[[B]], %[[EXT]] : vector<16xbf16>, vector<32xbf16>
  // CHECK:  %[[ME:.*]] = aievec.mul_elem %[[CC1]], %[[CC2]] : vector<32xbf16>, vector<32xbf16>, vector<16xf32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[ME]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  %1 = arith.extf %a : vector<16xbf16> to vector<16xf32>
  %2 = arith.extf %b : vector<16xbf16> to vector<16xf32>
  %3 = arith.mulf %1, %2 : vector<16xf32>
  return %3 : vector<16xf32>
}

// CHECK-LABEL: func @test_mul_elem_float
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xf32>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xf32>
func.func @test_mul_elem_float(%a : vector<16xf32>,
                         %b : vector<16xf32>) -> vector<16xf32> {
  // CHECK-NEXT:  %[[ME:.*]] = aievec.mul_elem %[[A]], %[[B]] : vector<16xf32>, vector<16xf32>, vector<16xf32>
  // CHECK-NEXT:  %[[CAST:.*]] = aievec.cast %[[ME]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  %3 = arith.mulf %a, %b : vector<16xf32>
  return %3 : vector<16xf32>
}

// CHECK-LABEL: func @test_i8_i16_mul_elem
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xi8>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<32xi16>
func.func @test_i8_i16_mul_elem(%a : vector<32xi8>, %b : vector<32xi16>) -> vector<32xi32> {
  // CHECK: %[[UNPACK:.*]] = aievec.unpack %arg0 : vector<32xi8>, vector<32xi16>
  // CHECK: %[[ME:.*]] = aievec.mul_elem %arg1, %[[UNPACK:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
  // CHECK: %[[CAST:.*]] = aievec.cast %[[ME]] {isResAcc = false} : vector<32xi32>, vector<32xi32>
  %1 = arith.extsi %b : vector<32xi16> to vector<32xi32>
  %2 = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %3 = arith.muli %1, %2 : vector<32xi32>
  return %3 : vector<32xi32>
}

// CHECK-LABEL: func @test_i8_i32_mul_elem
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi8>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xi32>
func.func @test_i8_i32_mul_elem(%a : vector<16xi8>, %b : vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[CC:.*]] = aievec.concat %arg0, %arg0 : vector<16xi8>, vector<32xi8>
  // CHECK: %[[UPS:.*]] = aievec.ups %[[CC]] {shift = 0 : i8} : vector<32xi8>, vector<32xi32>
  // CHECK: %[[CAST:.*]] = aievec.cast %[[UPS]] {isResAcc = false} : vector<32xi32>, vector<32xi32>
  // CHECK: %[[EXT:.*]] = aievec.ext %[[CAST]] {index = 0 : i8} : vector<32xi32>, vector<16xi32>
  // CHECK: %[[ME:.*]] = aievec.mul_elem %[[EXT]], %arg1 : vector<16xi32>, vector<16xi32>, vector<16xi64>
  // CHECK: %[[SRS:.*]] = aievec.srs %[[ME]], %[[C0]] : vector<16xi64>, i32, vector<16xi32>
  %1 = arith.extsi %a : vector<16xi8> to vector<16xi32>
  %2 = arith.muli %1, %b : vector<16xi32>
  return %2 : vector<16xi32>
}

// CHECK-LABEL: func @test_i16_i32_mul_elem
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xi32>
func.func @test_i16_i32_mul_elem(%a : vector<16xi16>, %b : vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[UPS:.*]] = aievec.ups %arg0 {shift = 0 : i8} : vector<16xi16>, vector<16xi32>
  // CHECK: %[[CAST:.*]] = aievec.cast %[[UPS]] {isResAcc = false} : vector<16xi32>, vector<16xi32>
  // CHECK: %[[ME:.*]] = aievec.mul_elem %[[CAST]], %arg1 : vector<16xi32>, vector<16xi32>, vector<16xi64>
  // CHECK: %[[SRS:.*]] = aievec.srs %[[ME]], %[[C0]] : vector<16xi64>, i32, vector<16xi32>
  %1 = arith.extsi %a : vector<16xi16> to vector<16xi32>
  %2 = arith.muli %1, %b : vector<16xi32>
  return %2 : vector<16xi32>
}
