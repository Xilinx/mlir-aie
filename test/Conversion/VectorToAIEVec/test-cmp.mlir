// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s

// CHECK-LABEL:func @veccmp_i32
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @veccmp_i32(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sgt"} : vector<16xi32>, vector<16xi32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpi sgt, %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1> 
}

// CHECK-LABEL:func @veccmp_eq
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @veccmp_eq(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "eq"} : vector<16xi32>, vector<16xi32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpi eq, %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_ne
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @veccmp_ne(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "ne"} : vector<16xi32>, vector<16xi32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpi ne, %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_lt
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @veccmp_lt(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "slt"} : vector<16xi32>, vector<16xi32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpi slt, %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_le
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @veccmp_le(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sle"} : vector<16xi32>, vector<16xi32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpi sle, %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_ge
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @veccmp_ge(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sge"} : vector<16xi32>, vector<16xi32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpi sge, %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_i16
// CHECK-SAME: %[[LHS:.*]]: vector<32xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi16>)
func.func @veccmp_i16(%arg0: vector<32xi16>, %arg1: vector<32xi16>) -> vector<32xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sgt"} : vector<32xi16>, vector<32xi16>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<32xi1>
  %0 = arith.cmpi sgt, %arg0, %arg1 : vector<32xi16>
  // CHECK: return %[[RES]] : vector<32xi1>
  return %0 : vector<32xi1>
}

// CHECK-LABEL:func @veccmp_i8
// CHECK-SAME: %[[LHS:.*]]: vector<64xi8>,
// CHECK-SAME: %[[RHS:.*]]: vector<64xi8>)
func.func @veccmp_i8(%arg0: vector<64xi8>, %arg1: vector<64xi8>) -> vector<64xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sgt"} : vector<64xi8>, vector<64xi8>, ui64
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui64 to vector<64xi1>
  %0 = arith.cmpi sgt, %arg0, %arg1 : vector<64xi8>
  // CHECK: return %[[RES]] : vector<64xi1>
  return %0 : vector<64xi1>
}

// CHECK-LABEL:func @veccmp_bf16
// CHECK-SAME: %[[LHS:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xbf16>)
func.func @veccmp_bf16(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sgt"} : vector<32xbf16>, vector<32xbf16>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<32xi1>
  %0 = arith.cmpf ogt, %arg0, %arg1 : vector<32xbf16>
  // CHECK: return %[[RES]] : vector<32xi1>
  return %0 : vector<32xi1>
}

// CHECK-LABEL:func @veccmp_f32
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @veccmp_f32(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sgt"} : vector<16xf32>, vector<16xf32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpf ogt, %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_feq
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @veccmp_feq(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "eq"} : vector<16xf32>, vector<16xf32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpf oeq, %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_fne
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @veccmp_fne(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "ne"} : vector<16xf32>, vector<16xf32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpf one, %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_flt
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @veccmp_flt(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "slt"} : vector<16xf32>, vector<16xf32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpf olt, %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_fle
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @veccmp_fle(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sle"} : vector<16xf32>, vector<16xf32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpf ole, %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}

// CHECK-LABEL:func @veccmp_fge
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @veccmp_fge(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xi1> {
  // CHECK: %[[CMP:.*]] = aievec.cmp %[[LHS]], %[[RHS]] {pred = "sge"} : vector<16xf32>, vector<16xf32>, ui32
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to vector<16xi1>
  %0 = arith.cmpf oge, %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xi1>
  return %0 : vector<16xi1>
}
