// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s

// CHECK-LABEL: func @vecaddi_i32(
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecaddi_i32(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[RES:.*]] = aievec.add_elem %[[LHS]], %[[RHS]] : vector<16xi32>
  %0 = arith.addi %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: func @vecaddi_32xi32(
// CHECK-SAME: %[[LHS:.*]]: vector<32xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi32>)
func.func @vecaddi_32xi32(%arg0: vector<32xi32>, %arg1: vector<32xi32>) -> vector<32xi32> {
  // CHECK:  %[[LCAST:.*]] = aievec.cast %[[LHS]] {isResAcc = true} : vector<32xi32>, vector<32xi32>
  // CHECK:  %[[RCAST:.*]] = aievec.cast %[[RHS]] {isResAcc = true} : vector<32xi32>, vector<32xi32>
  // CHECK:  %[[ADD:.*]] = aievec.add_elem %[[LCAST]], %[[RCAST]] : vector<32xi32>
  // CHECK:  %[[RES:.*]] = aievec.cast %[[ADD]] {isResAcc = false} : vector<32xi32>, vector<32xi32>
  %0 = arith.addi %arg0, %arg1 : vector<32xi32>
  // CHECK: return %[[RES]] : vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: func @vecaddi_i16_i32(
// CHECK-SAME: %[[LHS:.*]]: vector<32xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi16>)
func.func @vecaddi_i16_i32(%arg0 : vector<32xi16>, %arg1 : vector<32xi16>) -> vector<32xi32> {
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<32xi16>, vector<32xi32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<32xi16>, vector<32xi32>
  // CHECK:  %[[ADD:.*]] = aievec.add_elem %[[LUPS]], %[[RUPS]] : vector<32xi32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[ADD]] {isResAcc = false} : vector<32xi32>, vector<32xi32>
  %1 = arith.extsi %arg0 : vector<32xi16> to vector<32xi32>
  %2 = arith.extsi %arg1 : vector<32xi16> to vector<32xi32>
  %3 = arith.addi %1, %2 : vector<32xi32>
  // CHECK: return %[[CAST]] : vector<32xi32>
  return %3 : vector<32xi32>
}

// CHECK-LABEL: func @vecaddi_i16_i32_2(
// CHECK-SAME: %[[LHS:.*]]: vector<16xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecaddi_i16_i32_2(%arg0 : vector<16xi16>, %arg1 : vector<16xi32>) -> vector<16xi32> {
  // CHECK:  %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<16xi16>, vector<16xi64>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
  // CHECK:  %[[ADD:.*]] = aievec.add_elem %[[LUPS]], %[[RUPS]] : vector<16xi64>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[ADD]], %[[C0]] : vector<16xi64>, i32, vector<16xi32>
  %1 = arith.extsi %arg0 : vector<16xi16> to vector<16xi32>
  %2 = arith.addi %1, %arg1 : vector<16xi32>
  // CHECK: return %[[SRS]] : vector<16xi32>
  return %2 : vector<16xi32>
}

// CHECK-LABEL: func @vecaddi_i8_i32(
// CHECK-SAME: %[[LHS:.*]]: vector<32xi8>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi8>)
func.func @vecaddi_i8_i32(%arg0 : vector<32xi8>, %arg1 : vector<32xi8>) -> vector<32xi32> {
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<32xi8>, vector<32xi32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<32xi8>, vector<32xi32>
  // CHECK:  %[[ADD:.*]] = aievec.add_elem %[[LUPS]], %[[RUPS]] : vector<32xi32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[ADD]] {isResAcc = false} : vector<32xi32>, vector<32xi32>
  %1 = arith.extsi %arg0 : vector<32xi8> to vector<32xi32>
  %2 = arith.extsi %arg1 : vector<32xi8> to vector<32xi32>
  %3 = arith.addi %1, %2 : vector<32xi32>
  // CHECK: return %[[CAST]] : vector<32xi32>
  return %3 : vector<32xi32>
}

// CHECK-LABEL: func @vecaddi_i16(
// CHECK-SAME: %[[LHS:.*]]: vector<32xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi16>)
func.func @vecaddi_i16(%arg0: vector<32xi16>, %arg1: vector<32xi16>) -> vector<32xi16> {
  // CHECK: %[[RES:.*]] = aievec.add_elem %[[LHS]], %[[RHS]] : vector<32xi16>
  %0 = arith.addi %arg0, %arg1 : vector<32xi16>
  // CHECK: return %[[RES]] : vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL: func @vecaddi_i8(
// CHECK-SAME: %[[LHS:.*]]: vector<64xi8>,
// CHECK-SAME: %[[RHS:.*]]: vector<64xi8>)
func.func @vecaddi_i8(%arg0: vector<64xi8>, %arg1: vector<64xi8>) -> vector<64xi8> {
  // CHECK: %[[RES:.*]] = aievec.add_elem %[[LHS]], %[[RHS]] : vector<64xi8>
  %0 = arith.addi %arg0, %arg1 : vector<64xi8>
  // CHECK: return %[[RES]] : vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: func @vecsubi_i32(
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>)
func.func @vecsubi_i32(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[RES:.*]] = aievec.sub_elem %[[LHS]], %[[RHS]] : vector<16xi32>
  %0 = arith.subi %arg0, %arg1 : vector<16xi32>
  // CHECK: return %[[RES]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: func @vecsubi_i16(
// CHECK-SAME: %[[LHS:.*]]: vector<32xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi16>)
func.func @vecsubi_i16(%arg0: vector<32xi16>, %arg1: vector<32xi16>) -> vector<32xi16> {
  // CHECK: %[[RES:.*]] = aievec.sub_elem %[[LHS]], %[[RHS]] : vector<32xi16>
  %0 = arith.subi %arg0, %arg1 : vector<32xi16>
  // CHECK: return %[[RES]] : vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL: func @vecsubi_i8(
// CHECK-SAME: %[[LHS:.*]]: vector<64xi8>,
// CHECK-SAME: %[[RHS:.*]]: vector<64xi8>)
func.func @vecsubi_i8(%arg0: vector<64xi8>, %arg1: vector<64xi8>) -> vector<64xi8> {
  // CHECK: %[[RES:.*]] = aievec.sub_elem %[[LHS]], %[[RHS]] : vector<64xi8>
  %0 = arith.subi %arg0, %arg1 : vector<64xi8>
  // CHECK: return %[[RES]] : vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: func @vecsubi_32xi32(
// CHECK-SAME: %[[LHS:.*]]: vector<32xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi32>)
func.func @vecsubi_32xi32(%arg0: vector<32xi32>, %arg1: vector<32xi32>) -> vector<32xi32> {
  // CHECK:  %[[LCAST:.*]] = aievec.cast %[[LHS]] {isResAcc = true} : vector<32xi32>, vector<32xi32>
  // CHECK:  %[[RCAST:.*]] = aievec.cast %[[RHS]] {isResAcc = true} : vector<32xi32>, vector<32xi32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LCAST]], %[[RCAST]] : vector<32xi32>
  // CHECK:  %[[RES:.*]] = aievec.cast %[[SUB]] {isResAcc = false} : vector<32xi32>, vector<32xi32>
  %0 = arith.subi %arg0, %arg1 : vector<32xi32>
  // CHECK: return %[[RES]] : vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: func @vecsubi_i16_i32(
// CHECK-SAME: %[[LHS:.*]]: vector<32xi16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi16>)
func.func @vecsubi_i16_i32(%arg0 : vector<32xi16>, %arg1 : vector<32xi16>) -> vector<32xi32> {
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<32xi16>, vector<32xi32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<32xi16>, vector<32xi32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LUPS]], %[[RUPS]] : vector<32xi32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[SUB]] {isResAcc = false} : vector<32xi32>, vector<32xi32>
  %1 = arith.extsi %arg0 : vector<32xi16> to vector<32xi32>
  %2 = arith.extsi %arg1 : vector<32xi16> to vector<32xi32>
  %3 = arith.subi %1, %2 : vector<32xi32>
  // CHECK: return %[[CAST]] : vector<32xi32>
  return %3 : vector<32xi32>
}

// CHECK-LABEL: func @vecsubi_i8_i32(
// CHECK-SAME: %[[LHS:.*]]: vector<32xi8>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xi8>)
func.func @vecsubi_i8_i32(%arg0 : vector<32xi8>, %arg1 : vector<32xi8>) -> vector<32xi32> {
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<32xi8>, vector<32xi32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<32xi8>, vector<32xi32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LUPS]], %[[RUPS]] : vector<32xi32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[SUB]] {isResAcc = false} : vector<32xi32>, vector<32xi32>
  %1 = arith.extsi %arg0 : vector<32xi8> to vector<32xi32>
  %2 = arith.extsi %arg1 : vector<32xi8> to vector<32xi32>
  %3 = arith.subi %1, %2 : vector<32xi32>
  // CHECK: return %[[CAST]] : vector<32xi32>
  return %3 : vector<32xi32>
}

// CHECK-LABEL: func @vecaddf_f32(
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecaddf_f32(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK:  %[[LCAST:.*]] = aievec.cast %[[LHS]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK:  %[[RCAST:.*]] = aievec.cast %[[RHS]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.add_elem %[[LCAST]], %[[RCAST:.*]] : vector<16xf32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[SUB]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  %0 = arith.addf %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[CAST]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @vecaddf_bf16(
// CHECK-SAME: %[[LHS:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xbf16>)
func.func @vecaddf_bf16(%arg0: vector<16xbf16>, %arg1: vector<16xbf16>) -> vector<16xbf16> {
  // CHECK:  %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.add_elem %[[LUPS]], %[[RUPS]] : vector<16xf32>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[SUB]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  %0 = arith.addf %arg0, %arg1 : vector<16xbf16>
  // CHECK: return %[[SRS]] : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: func @vecsubf_f32(
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecsubf_f32(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK:  %[[LCAST:.*]] = aievec.cast %[[LHS]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK:  %[[RCAST:.*]] = aievec.cast %[[RHS]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LCAST]], %[[RCAST:.*]] : vector<16xf32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[SUB]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  %0 = arith.subf %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[CAST]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @vecsubf_bf16(
// CHECK-SAME: %[[LHS:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xbf16>)
func.func @vecsubf_bf16(%arg0: vector<16xbf16>, %arg1: vector<16xbf16>) -> vector<16xbf16> {
  // CHECK:  %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LUPS]], %[[RUPS]] : vector<16xf32>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[SUB]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  %0 = arith.subf %arg0, %arg1 : vector<16xbf16>
  // CHECK: return %[[SRS]] : vector<16xbf16>
  return %0 : vector<16xbf16>
}

