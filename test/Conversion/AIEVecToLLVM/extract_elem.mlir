// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm | FileCheck %s

func.func @i8_extract_elem(%arg0 : vector<64xi8>, %index : i32) -> i8 {
  %0 = aievec.ext_elem %arg0, %index : vector<64xi8>, i32, i8
  return %0 : i8
}

// CHECK-LABEL: @i8_extract_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>,
// CHECK-SAME: %[[INDEX:.*]]: i32
// CHECK: %[[CST:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VEXTELEM:.*]] = "xllvm.intr.aie2.vextract.elem8.I512"(
// CHECK-SAME: %[[ARG0]], %[[INDEX]], %[[CST]]) : 
// CHECK-SAME: (vector<64xi8>, i32, i32) -> i32
// CHECK-NEXT: %[[RES:.*]] = llvm.trunc %[[VEXTELEM]] : i32 to i8
// CHECK-NEXT: return %[[RES]] : i8

// -----

func.func @i16_extract_elem(%arg0 : vector<32xi16>, %index : i32) -> i16 {
  %0 = aievec.ext_elem %arg0, %index : vector<32xi16>, i32, i16
  return %0 : i16
}

// CHECK-LABEL: @i16_extract_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>,
// CHECK-SAME: %[[INDEX:.*]]: i32
// CHECK: %[[CST:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VEXTELEM:.*]] = "xllvm.intr.aie2.vextract.elem16.I512"(
// CHECK-SAME: %[[ARG0]], %[[INDEX]], %[[CST]]) : 
// CHECK-SAME: (vector<32xi16>, i32, i32) -> i32
// CHECK-NEXT: %[[RES:.*]] = llvm.trunc %[[VEXTELEM]] : i32 to i16
// CHECK-NEXT: return %[[RES]] : i16

// -----

func.func @i32_extract_elem(%arg0 : vector<16xi32>, %index : i32) -> i32 {
  %0 = aievec.ext_elem %arg0, %index : vector<16xi32>, i32, i32
  return %0 : i32
}

// CHECK-LABEL: @i32_extract_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>,
// CHECK-SAME: %[[INDEX:.*]]: i32
// CHECK: %[[CST:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[VEXTELEM:.*]] = "xllvm.intr.aie2.vextract.elem32.I512"(
// CHECK-SAME: %[[ARG0]], %[[INDEX]], %[[CST]]) : 
// CHECK-SAME: (vector<16xi32>, i32, i32) -> i32
// CHECK-NEXT: %[[RES:.*]] = llvm.trunc %[[VEXTELEM]] : i32 to i32
// CHECK-NEXT: return %[[RES]] : i32

// -----

func.func @bf16_extract_elem(%arg0 : vector<32xbf16>, %index : i32) -> bf16 {
  %0 = aievec.ext_elem %arg0, %index : vector<32xbf16>, i32, bf16
  return %0 : bf16
}

// CHECK-LABEL: @bf16_extract_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[INDEX:.*]]: i32
// CHECK: %[[CST:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG0]] : vector<32xbf16> to vector<32xi16>
// CHECK-NEXT: %[[VEXTELEM:.*]] = "xllvm.intr.aie2.vextract.elem16.I512"(
// CHECK-SAME: %[[BITCAST]], %[[INDEX]], %[[CST]]) : 
// CHECK-SAME: (vector<32xi16>, i32, i32) -> i32
// CHECK-NEXT: %[[TRUNC:.*]] = llvm.trunc %[[VEXTELEM]] : i32 to i16
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[TRUNC]] : i16 to bf16
// CHECK-NEXT: return %[[RES]] : bf16

// -----

func.func @f32_extract_elem(%arg0 : vector<16xf32>, %index : i32) -> f32 {
  %0 = aievec.ext_elem %arg0, %index : vector<16xf32>, i32, f32
  return %0 : f32
}

// CHECK-LABEL: @f32_extract_elem
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf32>,
// CHECK-SAME: %[[INDEX:.*]]: i32
// CHECK: %[[CST:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<16xi32>
// CHECK-NEXT: %[[VEXTELEM:.*]] = "xllvm.intr.aie2.vextract.elem32.I512"(
// CHECK-SAME: %[[BITCAST]], %[[INDEX]], %[[CST]]) : 
// CHECK-SAME: (vector<16xi32>, i32, i32) -> i32
// CHECK-NEXT: %[[TRUNC:.*]] = llvm.trunc %[[VEXTELEM]] : i32 to i32
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[TRUNC]] : i32 to f32
// CHECK-NEXT: return %[[RES]] : f32
