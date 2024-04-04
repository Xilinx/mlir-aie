// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm | FileCheck %s

func.func @i8_concat(%arg0 : vector<32xi8>, %arg1 : vector<32xi8>) -> vector<64xi8> {
  %0 = aievec.concat %arg0, %arg1 : vector<32xi8>, vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @i8_concat
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi8>,
// CHECK-SAME: %[[ARG1:.*]]: vector<32xi8>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi8> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<32xi8> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I512.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: return %[[RES]] : vector<64xi8>

// -----

func.func @bf16_concat(%arg0 : vector<16xbf16>, %arg1 : vector<16xbf16>) -> vector<32xbf16> {
  %0 = aievec.concat %arg0, %arg1 : vector<16xbf16>, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @bf16_concat
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[ARG1:.*]]: vector<16xbf16>
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I512.I256"(
// CHECK-SAME: %[[BITCAST0]], %[[BITCAST1]]) : 
// CHECK-SAME: (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<16xi32> to vector<32xbf16>
// CHECK-NEXT: return %[[RES]] : vector<32xbf16>

