// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s

// CHECK-LABEL:func @vecsel_i32
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecsel_i32(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sgt"} : vector<16xi32>, vector<16xi32>, ui32
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<16xi32>, vector<16xi32>, ui32, vector<16xi32>
  %0 = arith.cmpi sgt, %arg0, %arg1 : vector<16xi32>
  %1 = arith.select %0, %arg0, %arg1 : vector<16xi1>, vector<16xi32>
  // CHECK: return %[[SEL]] : vector<16xi32>
  return %1 : vector<16xi32> 
}

// CHECK-LABEL:func @vecsel_i32_unsigned_cmp
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecsel_i32_unsigned_cmp(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "ugt"} : vector<16xi32>, vector<16xi32>, ui32
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<16xi32>, vector<16xi32>, ui32, vector<16xi32>
  %0 = arith.cmpi ugt, %arg0, %arg1 : vector<16xi32>
  %1 = arith.select %0, %arg0, %arg1 : vector<16xi1>, vector<16xi32>
  // CHECK: return %[[SEL]] : vector<16xi32>
  return %1 : vector<16xi32>
}

// CHECK-LABEL:func @vecsel_i16
// CHECK-SAME: %[[LHS:.*]]: vector<32xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi16>)
func.func @vecsel_i16(%arg0: vector<32xi16>, %arg1: vector<32xi16>) -> vector<32xi16> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "slt"} : vector<32xi16>, vector<32xi16>, ui32
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<32xi16>, vector<32xi16>, ui32, vector<32xi16>
  %0 = arith.cmpi slt, %arg0, %arg1 : vector<32xi16>
  %1 = arith.select %0, %arg0, %arg1 : vector<32xi1>, vector<32xi16>
  // CHECK: return %[[SEL]] : vector<32xi16>
  return %1 : vector<32xi16>
}

// CHECK-LABEL:func @vecsel_i16_unsigned_cmp
// CHECK-SAME: %[[LHS:.*]]: vector<32xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi16>)
func.func @vecsel_i16_unsigned_cmp(%arg0: vector<32xi16>, %arg1: vector<32xi16>) -> vector<32xi16> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "ult"} : vector<32xi16>, vector<32xi16>, ui32
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<32xi16>, vector<32xi16>, ui32, vector<32xi16>
  %0 = arith.cmpi ult, %arg0, %arg1 : vector<32xi16>
  %1 = arith.select %0, %arg0, %arg1 : vector<32xi1>, vector<32xi16>
  // CHECK: return %[[SEL]] : vector<32xi16>
  return %1 : vector<32xi16>
}

// CHECK-LABEL:func @vecsel_i8
// CHECK-SAME: %[[LHS:.*]]: vector<64xi8>,
// CHECK-SAME: %[[RHS:.*]]: vector<64xi8>)
func.func @vecsel_i8(%arg0: vector<64xi8>, %arg1: vector<64xi8>) -> vector<64xi8> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sge"} : vector<64xi8>, vector<64xi8>, ui64
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<64xi8>, vector<64xi8>, ui64, vector<64xi8>
  %0 = arith.cmpi sge, %arg0, %arg1 : vector<64xi8>
  %1 = arith.select %0, %arg0, %arg1 : vector<64xi1>, vector<64xi8>
  // CHECK: return %[[SEL]] : vector<64xi8>
  return %1 : vector<64xi8>
}

// CHECK-LABEL:func @vecsel_i8_unsigned_cmp
// CHECK-SAME: %[[LHS:.*]]: vector<64xi8>,
// CHECK-SAME: %[[RHS:.*]]: vector<64xi8>)
func.func @vecsel_i8_unsigned_cmp(%arg0: vector<64xi8>, %arg1: vector<64xi8>) -> vector<64xi8> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "uge"} : vector<64xi8>, vector<64xi8>, ui64
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<64xi8>, vector<64xi8>, ui64, vector<64xi8>
  %0 = arith.cmpi uge, %arg0, %arg1 : vector<64xi8>
  %1 = arith.select %0, %arg0, %arg1 : vector<64xi1>, vector<64xi8>
  // CHECK: return %[[SEL]] : vector<64xi8>
  return %1 : vector<64xi8>
}

// CHECK-LABEL:func @vecsel_bf16
// CHECK-SAME: %[[LHS:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xbf16>)
func.func @vecsel_bf16(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sle"} : vector<32xbf16>, vector<32xbf16>, ui32
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<32xbf16>, vector<32xbf16>, ui32, vector<32xbf16>
  %0 = arith.cmpf ole, %arg0, %arg1 : vector<32xbf16>
  %1 = arith.select %0, %arg0, %arg1 : vector<32xi1>, vector<32xbf16>
  // CHECK: return %[[SEL]] : vector<32xbf16>
  return %1 : vector<32xbf16>
}

// CHECK-LABEL:func @vecsel_bf16_unsigned_cmp
// CHECK-SAME: %[[LHS:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xbf16>)
func.func @vecsel_bf16_unsigned_cmp(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "ule"} : vector<32xbf16>, vector<32xbf16>, ui32
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<32xbf16>, vector<32xbf16>, ui32, vector<32xbf16>
  %0 = arith.cmpf ule, %arg0, %arg1 : vector<32xbf16>
  %1 = arith.select %0, %arg0, %arg1 : vector<32xi1>, vector<32xbf16>
  // CHECK: return %[[SEL]] : vector<32xbf16>
  return %1 : vector<32xbf16>
}

// CHECK-LABEL:func @vecsel_f32
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecsel_f32(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sgt"} : vector<16xf32>, vector<16xf32>, ui32
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<16xf32>, vector<16xf32>, ui32, vector<16xf32>
  %0 = arith.cmpf ogt, %arg0, %arg1 : vector<16xf32>
  %1 = arith.select %0, %arg0, %arg1 : vector<16xi1>, vector<16xf32>
  // CHECK: return %[[SEL]] : vector<16xf32>
  return %1 : vector<16xf32>
}

// CHECK-LABEL:func @vecsel_f32_unsigned_cmp
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecsel_f32_unsigned_cmp(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "ugt"} : vector<16xf32>, vector<16xf32>, ui32
  // CHECK: %[[SEL:.*]] = aievec.sel %[[LHS]], %[[RHS]], %[[CMP]] : vector<16xf32>, vector<16xf32>, ui32, vector<16xf32>
  %0 = arith.cmpf ugt, %arg0, %arg1 : vector<16xf32>
  %1 = arith.select %0, %arg0, %arg1 : vector<16xi1>, vector<16xf32>
  // CHECK: return %[[SEL]] : vector<16xf32>
  return %1 : vector<16xf32>
}
