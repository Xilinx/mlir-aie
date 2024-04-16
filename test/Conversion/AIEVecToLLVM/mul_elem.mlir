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

func.func @i32_i32_i32_mul_elem(%arg0 : vector<16xi32>, %arg1 : vector<16xi32>) -> vector<16xi64> {
  %0 = aievec.mul_elem %arg0, %arg1 : vector<16xi32>, vector<16xi32>, vector<16xi64>
  return %0 : vector<16xi64>
}

// CHECK-LABEL: @i32_i32_i32_mul_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>,
// CHECK-SAME: %[[ARG1:.*]]: vector<16xi32>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VBROADCAST:.*]] = "xllvm.intr.aie2.vbroadcast32.I512"(%[[CST0]]) : (i32) -> vector<16xi32>
// CHECK-NEXT: %[[UNDEF:.*]]  = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[SHUFF0:.*]] = "xllvm.intr.aie2.vshuffle"(%[[ARG0]], %[[VBROADCAST]], %[[CST1]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[CST2:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[SHUFF1:.*]] = "xllvm.intr.aie2.vshuffle"(%[[ARG0]], %[[VBROADCAST]], %[[CST2]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[CST3:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[SHUFF2:.*]] = "xllvm.intr.aie2.vshuffle"(%[[ARG1]], %[[UNDEF]], %[[CST3]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[CST4:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[SHUFF3:.*]] = "xllvm.intr.aie2.vshuffle"(%[[ARG1]], %[[UNDEF]], %[[CST4]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[CST5:.*]] = llvm.mlir.constant(858 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[SHUFF1]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: %[[ACC0:.*]] = "xllvm.intr.aie2.I512.I512.acc64.mul.conf"(%[[BITCAST0]], %[[SHUFF3]], %[[CST5]]) : (vector<64xi8>, vector<16xi32>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[CST6:.*]] = llvm.mlir.constant(1626 : i32) : i32
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[SHUFF1]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: %[[ACC1:.*]] = "xllvm.intr.aie2.I512.I512.ACC1024.acc64.mac.conf"(%[[BITCAST1]], %[[SHUFF2]], %[[ACC0]], %[[CST6]]) : (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[CST7:.*]] = llvm.mlir.constant(346 : i32) : i32
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[SHUFF0]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: %[[ACC2:.*]] = "xllvm.intr.aie2.I512.I512.ACC1024.acc64.mac.conf"(%[[BITCAST2]], %[[SHUFF3]], %[[ACC1]], %[[CST7]]) : (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[CST8:.*]] = llvm.mlir.constant(1114 : i32) : i32
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[SHUFF0]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: %[[ACC3:.*]] = "xllvm.intr.aie2.I512.I512.ACC1024.acc64.mac.conf"(%[[BITCAST3]], %[[SHUFF2]], %[[ACC2]], %[[CST8]]) : (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32) -> vector<16xi64>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[ACC3]] : vector<16xi64> to vector<16xi64>
// CHECK-NEXT: return %[[RES]] : vector<16xi64>

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