// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm | FileCheck %s

func.func @i16_i16_i32_mul_elem(%arg0 : vector<32xi16>, %arg1 : vector<32xi16>) -> vector<32xi32> {
  %0 = aievec.mul_elem %arg0, %arg1 : vector<32xi16>, vector<32xi16>, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @i16_i16_i32_mul_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>,
// CHECK-SAME: %[[ARG1:.*]]: vector<32xi16>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(824 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi16> to vector<64xi8>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<32xi16> to vector<16xi32>
// CHECK-NEXT: %[[MULCONF:.*]] = "xllvm.intr.aie2.I512.I512.acc32.mul.conf"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]], %[[CST]]) : 
// CHECK-SAME: (vector<64xi8>, vector<16xi32>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[MULCONF]] : vector<16xi64> to vector<32xi32>
// CHECK-NEXT: return %[[RES]] : vector<32xi32>

// -----

func.func @i8_i8_i32_mul_elem(%arg0 : vector<64xi8>, %arg1 : vector<64xi8>) -> vector<32xi32> {
  %0 = aievec.mul_elem %arg0, %arg1 : vector<64xi8>, vector<64xi8>, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @i8_i8_i32_mul_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>,
// CHECK-SAME: %[[ARG1:.*]]: vector<64xi8>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(808 : i32) : i32
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<64xi8> to vector<16xi32>
// CHECK-NEXT: %[[MULCONF:.*]] = "xllvm.intr.aie2.I512.I512.acc32.mul.conf"(
// CHECK-SAME: %[[ARG0]], %[[BITCAST1]], %[[CST]]) : 
// CHECK-SAME: (vector<64xi8>, vector<16xi32>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[MULCONF]] : vector<16xi64> to vector<32xi32>
// CHECK-NEXT: return %[[RES]] : vector<32xi32>

// -----

func.func @bf16_bf16_f32_mul_elem(%arg0 : vector<32xbf16>, %arg1 : vector<32xbf16>) -> vector<16xf32> {
  %0 = aievec.mul_elem %arg0, %arg1 : vector<32xbf16>, vector<32xbf16>, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @bf16_bf16_f32_mul_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[ARG1:.*]]: vector<32xbf16>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(60 : i32) : i32
// CHECK-NEXT: %[[MULCONF:.*]] = "xllvm.intr.aie2.bf.mul16.conf"(
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[CST]]) : 
// CHECK-SAME: (vector<32xbf16>, vector<32xbf16>, i32) -> vector<8xi64>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[MULCONF]] : vector<8xi64> to vector<16xf32>
// CHECK-NEXT: return %[[RES]] : vector<16xf32>