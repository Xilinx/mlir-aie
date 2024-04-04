// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm -cse | FileCheck %s

func.func @i8_ext(%arg0 : vector<64xi8>) -> (vector<32xi8>, vector<32xi8>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xi8>, vector<32xi8>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<64xi8>, vector<32xi8>
  return %0, %1 : vector<32xi8>, vector<32xi8>
}

// CHECK-LABEL: @i8_ext
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xi8> to vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<8xi32> to vector<32xi8>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<8xi32> to vector<32xi8>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<32xi8>, vector<32xi8>

// -----

func.func @bf16_ext(%arg0 : vector<32xbf16>) -> (vector<16xbf16>, vector<16xbf16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
  return %0, %1 : vector<16xbf16>, vector<16xbf16>
}

// CHECK-LABEL: @bf16_ext
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xbf16> to vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<8xi32> to vector<16xbf16>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<8xi32> to vector<16xbf16>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<16xbf16>, vector<16xbf16>