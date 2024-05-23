// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm | FileCheck %s

func.func @i8_max(%arg0 : vector<64xi8>) -> vector<64xi8> {
  %0 = aievec.max %arg0, %arg0 : vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @i8_max
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VMAX:.*]] = "xllvm.intr.aie2.vmax.lt8"(
// CHECK-SAME: %[[ARG0]], %[[ARG0]], %[[CST]]) : 
// CHECK-SAME: (vector<64xi8>, vector<64xi8>, i32) -> !llvm.struct<(vector<64xi8>, i32)>
// CHECK-NEXT: %[[RES:.*]] = llvm.extractvalue %[[VMAX]][0] : !llvm.struct<(vector<64xi8>, i32)>
// CHECK-NEXT: return %[[RES]] : vector<64xi8>

// -----

func.func @i16_max(%arg0 : vector<32xi16>) -> vector<32xi16> {
  %0 = aievec.max %arg0, %arg0 : vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL: @i16_max
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VMAX:.*]] = "xllvm.intr.aie2.vmax.lt16"(
// CHECK-SAME: %[[ARG0]], %[[ARG0]], %[[CST]]) : 
// CHECK-SAME: (vector<32xi16>, vector<32xi16>, i32) -> !llvm.struct<(vector<32xi16>, i32)>
// CHECK-NEXT: %[[RES:.*]] = llvm.extractvalue %[[VMAX]][0] : !llvm.struct<(vector<32xi16>, i32)>
// CHECK-NEXT: return %[[RES]] : vector<32xi16>

// -----

func.func @i32_max(%arg0 : vector<16xi32>) -> vector<16xi32> {
  %0 = aievec.max %arg0, %arg0 : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: @i32_max
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VMAX:.*]] = "xllvm.intr.aie2.vmax.lt32"(
// CHECK-SAME: %[[ARG0]], %[[ARG0]], %[[CST]]) : 
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32) -> !llvm.struct<(vector<16xi32>, i32)>
// CHECK-NEXT: %[[RES:.*]] = llvm.extractvalue %[[VMAX]][0] : !llvm.struct<(vector<16xi32>, i32)>
// CHECK-NEXT: return %[[RES]] : vector<16xi32>

// -----

func.func @bf16_max(%arg0 : vector<32xbf16>) -> vector<32xbf16> {
  %0 = aievec.max %arg0, %arg0 : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @bf16_max
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>
// CHECK-NEXT: %[[VMAX:.*]] = "xllvm.intr.aie2.vmax.ltbf16"(
// CHECK-SAME: %[[ARG0]], %[[ARG0]]) : 
// CHECK-SAME: (vector<32xbf16>, vector<32xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)>
// CHECK-NEXT: %[[RES:.*]] = llvm.extractvalue %[[VMAX]][0] : !llvm.struct<(vector<32xbf16>, i32)> 
// CHECK-NEXT: return %[[RES]] : vector<32xbf16>
