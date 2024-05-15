// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm -cse | FileCheck %s

func.func @v32i8_ext_v64i8(%arg0 : vector<64xi8>) -> (vector<32xi8>, vector<32xi8>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xi8>, vector<32xi8>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<64xi8>, vector<32xi8>
  return %0, %1 : vector<32xi8>, vector<32xi8>
}

// CHECK-LABEL: @v32i8_ext_v64i8
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

func.func @v64i8_ext_v128i8(%arg0 : vector<128xi8>) -> (vector<64xi8>, vector<64xi8>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<128xi8>, vector<64xi8>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<128xi8>, vector<64xi8>
  return %0, %1 : vector<64xi8>, vector<64xi8>
}

// CHECK-LABEL: @v64i8_ext_v128i8
// CHECK-SAME: %[[ARG0:.*]]: vector<128xi8>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<128xi8> to vector<32xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<16xi32> to vector<64xi8>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<64xi8>, vector<64xi8>

// -----

func.func @v32i8_ext_v128i8(%arg0 : vector<128xi8>) -> (vector<32xi8>, vector<32xi8>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<128xi8>, vector<32xi8>
  %1 = aievec.ext %arg0 {index = 3 : i8} : vector<128xi8>, vector<32xi8>
  return %0, %1 : vector<32xi8>, vector<32xi8>
}

// CHECK-LABEL: @v32i8_ext_v128i8
// CHECK-SAME: %[[ARG0:.*]]: vector<128xi8>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<128xi8> to vector<32xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<8xi32> to vector<32xi8>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<8xi32> to vector<32xi8>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<32xi8>, vector<32xi8>

// -----

func.func @v16i8_ext_v64i8(%arg0 : vector<64xi8>) -> (vector<16xi8>, vector<16xi8>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xi8>, vector<16xi8>
  %1 = aievec.ext %arg0 {index = 3 : i8} : vector<64xi8>, vector<16xi8>
  return %0, %1 : vector<16xi8>, vector<16xi8>
}

// CHECK-LABEL: @v16i8_ext_v64i8
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>
// CHECK: %[[UNDEF0:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xi8> to vector<16xi32>
// CHECK-NEXT: %[[VSHIFT0:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[UNDEF0]], %[[CST0]], %[[CST0]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT0]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<4xi32> to vector<16xi8>
// CHECK-NEXT: %[[UNDEF1:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST48:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK-NEXT: %[[VSHIFT1:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[UNDEF1]], %[[CST0]], %[[CST48]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT1]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<4xi32> to vector<16xi8>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<16xi8>, vector<16xi8>

// -----

func.func @v16i16_ext_v32i16(%arg0 : vector<32xi16>) -> (vector<16xi16>, vector<16xi16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<32xi16>, vector<16xi16>
  return %0, %1 : vector<16xi16>, vector<16xi16>
}

// CHECK-LABEL: @v16i16_ext_v32i16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi16> to vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<8xi32> to vector<16xi16>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<8xi32> to vector<16xi16>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<16xi16>, vector<16xi16>

// -----

func.func @v32i16_ext_v64i16(%arg0 : vector<64xi16>) -> (vector<32xi16>, vector<32xi16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xi16>, vector<32xi16>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<64xi16>, vector<32xi16>
  return %0, %1 : vector<32xi16>, vector<32xi16>
}

// CHECK-LABEL: @v32i16_ext_v64i16
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi16>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xi16> to vector<32xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<16xi32> to vector<32xi16>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<16xi32> to vector<32xi16>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<32xi16>, vector<32xi16>

// -----

func.func @v16i16_ext_v64i16(%arg0 : vector<64xi16>) -> (vector<16xi16>, vector<16xi16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xi16>, vector<16xi16>
  %1 = aievec.ext %arg0 {index = 3 : i8} : vector<64xi16>, vector<16xi16>
  return %0, %1 : vector<16xi16>, vector<16xi16>
}

// CHECK-LABEL: @v16i16_ext_v64i16
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi16>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xi16> to vector<32xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<8xi32> to vector<16xi16>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<8xi32> to vector<16xi16>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<16xi16>, vector<16xi16>

// -----

func.func @v8i16_ext_v32i16(%arg0 : vector<32xi16>) -> (vector<8xi16>, vector<8xi16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xi16>, vector<8xi16>
  %1 = aievec.ext %arg0 {index = 3 : i8} : vector<32xi16>, vector<8xi16>
  return %0, %1 : vector<8xi16>, vector<8xi16>
}

// CHECK-LABEL: @v8i16_ext_v32i16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>
// CHECK: %[[UNDEF0:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi16> to vector<16xi32>
// CHECK-NEXT: %[[VSHIFT0:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[UNDEF0]], %[[CST0]], %[[CST0]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT0]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<4xi32> to vector<8xi16>
// CHECK-NEXT: %[[UNDEF1:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST48:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK-NEXT: %[[VSHIFT1:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[UNDEF1]], %[[CST0]], %[[CST48]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT1]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<4xi32> to vector<8xi16>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<8xi16>, vector<8xi16>

// -----

func.func @v8i32_ext_v16i32(%arg0 : vector<16xi32>) -> (vector<8xi32>, vector<8xi32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<16xi32>, vector<8xi32>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<16xi32>, vector<8xi32>
  return %0, %1 : vector<8xi32>, vector<8xi32>
}

// CHECK-LABEL: @v8i32_ext_v16i32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[ARG0]], %[[CST0]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[ARG0]], %[[CST1]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: return %[[EXT0]], %[[EXT1]] : vector<8xi32>, vector<8xi32>

// -----

func.func @v16i32_ext_v32i32(%arg0 : vector<32xi32>) -> (vector<16xi32>, vector<16xi32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xi32>, vector<16xi32>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<32xi32>, vector<16xi32>
  return %0, %1 : vector<16xi32>, vector<16xi32>
}

// CHECK-LABEL: @v16i32_ext_v32i32
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi32>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[ARG0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[ARG0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: return %[[EXT0]], %[[EXT1]] : vector<16xi32>, vector<16xi32>

// -----

func.func @v8i32_ext_v32i32(%arg0 : vector<32xi32>) -> (vector<8xi32>, vector<8xi32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xi32>, vector<8xi32>
  %1 = aievec.ext %arg0 {index = 3 : i8} : vector<32xi32>, vector<8xi32>
  return %0, %1 : vector<8xi32>, vector<8xi32>
}

// CHECK-LABEL: @v8i32_ext_v32i32
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi32>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[ARG0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[ARG0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: return %[[EXT0]], %[[EXT1]] : vector<8xi32>, vector<8xi32>

// -----

func.func @v4i32_ext_v16i32(%arg0 : vector<16xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<16xi32>, vector<4xi32>
  %1 = aievec.ext %arg0 {index = 3 : i8} : vector<16xi32>, vector<4xi32>
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: @v4i32_ext_v16i32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>
// CHECK: %[[UNDEF0:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[VSHIFT0:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[ARG0]], %[[UNDEF0]], %[[CST0]], %[[CST0]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT0]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: %[[UNDEF1:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST48:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK-NEXT: %[[VSHIFT1:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[ARG0]], %[[UNDEF1]], %[[CST0]], %[[CST48]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT1]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: return %[[EXT0]], %[[EXT1]] : vector<4xi32>, vector<4xi32>

// -----

func.func @v16bf16_ext_v32bf16(%arg0 : vector<32xbf16>) -> (vector<16xbf16>, vector<16xbf16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
  return %0, %1 : vector<16xbf16>, vector<16xbf16>
}

// CHECK-LABEL: @v16bf16_ext_v32bf16
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

// -----

func.func @v32bf16_ext_v64bf16(%arg0 : vector<64xbf16>) -> (vector<32xbf16>, vector<32xbf16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xbf16>, vector<32xbf16>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<64xbf16>, vector<32xbf16>
  return %0, %1 : vector<32xbf16>, vector<32xbf16>
}

// CHECK-LABEL: @v32bf16_ext_v64bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<64xbf16>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xbf16> to vector<32xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<16xi32> to vector<32xbf16>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<16xi32> to vector<32xbf16>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<32xbf16>, vector<32xbf16>

// -----

func.func @v16bf16_ext_v64bf16(%arg0 : vector<64xbf16>) -> (vector<16xbf16>, vector<16xbf16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xbf16>, vector<16xbf16>
  %1 = aievec.ext %arg0 {index = 2 : i8} : vector<64xbf16>, vector<16xbf16>
  return %0, %1 : vector<16xbf16>, vector<16xbf16>
}

// CHECK-LABEL: @v16bf16_ext_v64bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<64xbf16>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xbf16> to vector<32xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<8xi32> to vector<16xbf16>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<8xi32> to vector<16xbf16>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<16xbf16>, vector<16xbf16>

// -----

func.func @v8bf16_ext_v32bf16(%arg0 : vector<32xbf16>) -> (vector<8xbf16>, vector<8xbf16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xbf16>, vector<8xbf16>
  %1 = aievec.ext %arg0 {index = 3 : i8} : vector<32xbf16>, vector<8xbf16>
  return %0, %1 : vector<8xbf16>, vector<8xbf16>
}

// CHECK-LABEL: @v8bf16_ext_v32bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>
// CHECK: %[[UNDEF0:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xbf16> to vector<16xi32>
// CHECK-NEXT: %[[VSHIFT0:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[UNDEF0]], %[[CST0]], %[[CST0]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT0]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<4xi32> to vector<8xbf16>
// CHECK-NEXT: %[[UNDEF1:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST48:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK-NEXT: %[[VSHIFT1:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[UNDEF1]], %[[CST0]], %[[CST48]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT1]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<4xi32> to vector<8xbf16>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<8xbf16>, vector<8xbf16>

// -----

func.func @v8f32_ext_v16f32(%arg0 : vector<16xf32>) -> (vector<8xf32>, vector<8xf32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<16xf32>, vector<8xf32>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<16xf32>, vector<8xf32>
  return %0, %1 : vector<8xf32>, vector<8xf32>
}

// CHECK-LABEL: @v8f32_ext_v16f32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf32>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<8xi32> to vector<8xf32>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<8xi32> to vector<8xf32>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<8xf32>, vector<8xf32>

// -----

func.func @v16f32_ext_v32f32(%arg0 : vector<32xf32>) -> (vector<16xf32>, vector<16xf32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xf32>, vector<16xf32>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<32xf32>, vector<16xf32>
  return %0, %1 : vector<16xf32>, vector<16xf32>
}

// CHECK-LABEL: @v16f32_ext_v32f32
// CHECK-SAME: %[[ARG0:.*]]: vector<32xf32>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<16xi32> to vector<16xf32>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<16xi32> to vector<16xf32>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<16xf32>, vector<16xf32>

// -----

func.func @v8f32_ext_v32f32(%arg0 : vector<32xf32>) -> (vector<8xf32>, vector<8xf32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xf32>, vector<8xf32>
  %1 = aievec.ext %arg0 {index = 2 : i8} : vector<32xf32>, vector<8xf32>
  return %0, %1 : vector<8xf32>, vector<8xf32>
}

// CHECK-LABEL: @v8f32_ext_v32f32
// CHECK-SAME: %[[ARG0:.*]]: vector<32xf32>
// CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST0]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<8xi32> to vector<8xf32>
// CHECK-NEXT: %[[CST1:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[CST1]]) : 
// CHECK-SAME: (vector<32xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<8xi32> to vector<8xf32>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<8xf32>, vector<8xf32>

// -----

func.func @v4f32_ext_v16f32(%arg0 : vector<16xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<16xf32>, vector<4xf32>
  %1 = aievec.ext %arg0 {index = 3 : i8} : vector<16xf32>, vector<4xf32>
  return %0, %1 : vector<4xf32>, vector<4xf32>
}

// CHECK-LABEL: @v4f32_ext_v16f32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf32>
// CHECK: %[[UNDEF0:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<16xi32>
// CHECK-NEXT: %[[VSHIFT0:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[UNDEF0]], %[[CST0]], %[[CST0]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT0]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: %[[RES0:.*]] = llvm.bitcast %[[EXT0]] : vector<4xi32> to vector<4xf32>
// CHECK-NEXT: %[[UNDEF1:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK-NEXT: %[[CST48:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK-NEXT: %[[VSHIFT1:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[UNDEF1]], %[[CST0]], %[[CST48]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.extract.I128.I512"(
// CHECK-SAME: %[[VSHIFT1]]) : 
// CHECK-SAME: (vector<16xi32>) -> vector<4xi32>
// CHECK-NEXT: %[[RES1:.*]] = llvm.bitcast %[[EXT1]] : vector<4xi32> to vector<4xf32>
// CHECK-NEXT: return %[[RES0]], %[[RES1]] : vector<4xf32>, vector<4xf32>
