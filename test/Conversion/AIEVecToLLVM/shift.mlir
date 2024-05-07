// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm | FileCheck %s

// -----

func.func @i8_shift(%arg0 : vector<64xi8>, %shift : i32) -> vector<64xi8> {
  %0 = aievec.shift %arg0, %arg0, %shift {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @i8_shift
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>,
// CHECK-SAME: %[[SHIFT:.*]]: i32
// CHECK: %[[CST:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xi8> to vector<16xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<64xi8> to vector<16xi32>
// CHECK-NEXT: %[[VSHIFT:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]], %[[CST]], %[[SHIFT]]) : 
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[VSHIFT]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: return %[[RES]] : vector<64xi8>

// -----

func.func @i16_shift(%arg0 : vector<32xi16>, %shift : i32) -> vector<32xi16> {
  %0 = aievec.shift %arg0, %arg0, %shift {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL: @i16_shift
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>,
// CHECK-SAME: %[[SHIFT:.*]]: i32
// CHECK: %[[CST:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi16> to vector<16xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi16> to vector<16xi32>
// CHECK-NEXT: %[[VSHIFT:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]], %[[CST]], %[[SHIFT]]) : 
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[VSHIFT]] : vector<16xi32> to vector<32xi16>
// CHECK-NEXT: return %[[RES]] : vector<32xi16>

// -----

func.func @i32_shift(%arg0 : vector<16xi32>, %shift : i32) -> vector<16xi32> {
  %0 = aievec.shift %arg0, %arg0, %shift {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: @i32_shift
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>,
// CHECK-SAME: %[[SHIFT:.*]]: i32
// CHECK: %[[CST:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VSHIFT:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[ARG0]], %[[ARG0]], %[[CST]], %[[SHIFT]]) : 
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[VSHIFT]] : vector<16xi32> to vector<16xi32>
// CHECK-NEXT: return %[[RES]] : vector<16xi32>

// -----

func.func @bf16_shift(%arg0 : vector<32xbf16>, %shift : i32) -> vector<32xbf16> {
  %0 = aievec.shift %arg0, %arg0, %shift {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @bf16_shift
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[SHIFT:.*]]: i32
// CHECK: %[[CST:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VSHIFT:.*]] = "xllvm.intr.aie2.vshift.bf512.bf512"(
// CHECK-SAME: %[[ARG0]], %[[ARG0]], %[[CST]], %[[SHIFT]]) : 
// CHECK-SAME: (vector<32xbf16>, vector<32xbf16>, i32, i32) -> vector<32xbf16>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[VSHIFT]] : vector<32xbf16> to vector<32xbf16>
// CHECK-NEXT: return %[[RES]] : vector<32xbf16>
