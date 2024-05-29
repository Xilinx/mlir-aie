// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm -verify-diagnostics | FileCheck %s

func.func @i8_max(%arg0 : vector<64xi8>) -> vector<64xi8> {
  %0 = aievec.max %arg0, %arg0 : vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @i8_max
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>
// CHECK: %[[CST:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VMAX:.*]] = "xllvm.intr.aie2.vmax.lt8"(
// CHECK-SAME: %[[ARG0]], %[[ARG0]], %[[CST]]) : 
// CHECK-SAME: (vector<64xi8>, vector<64xi8>, i32) -> !llvm.struct<(vector<64xi8>, vector<2xi32>)>
// CHECK-NEXT: %[[RES:.*]] = llvm.extractvalue %[[VMAX]][0] : !llvm.struct<(vector<64xi8>, vector<2xi32>)>
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

// -----

func.func @invalid_i4_max(%arg0 : vector<128xi4>) -> vector<128xi4> {
  // expected-warning @+2 {{aievec.max conversion fails due to unsupported element data type.}}
  // expected-error @+1 {{failed to legalize operation 'aievec.max' that was explicitly marked illegal}}
  %0 = aievec.max %arg0, %arg0 : vector<128xi4>
  return %0 : vector<128xi4>
}

// -----

func.func @invalid_i8_max(%arg0 : vector<128xi8>) -> vector<128xi8> {
  // expected-warning @+2 {{aievec.max conversion with 1024-bit result is not supported.}}
  // expected-error @+1 {{failed to legalize operation 'aievec.max' that was explicitly marked illegal}}
  %0 = aievec.max %arg0, %arg0 : vector<128xi8>
  return %0 : vector<128xi8>
}
