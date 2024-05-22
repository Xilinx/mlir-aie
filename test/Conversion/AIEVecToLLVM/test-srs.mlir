// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm | FileCheck %s

func.func @v32i16_srs_v32i32(%arg0 : vector<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<32xi32>, i32, vector<32xi16>
  %1 = aievec.srs %arg0, %c5 : vector<32xi32>, i32, vector<32xi16>
  return
}

// CHECK-LABEL: @v32i16_srs_v32i32
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi32>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi32> to vector<16xi64>
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.I512.v32.acc32.srs"(
// CHECK-SAME: [[BITCAST0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<32xi16>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi32> to vector<16xi64>
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.I512.v32.acc32.srs"(
// CHECK-SAME: [[BITCAST1]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<32xi16>

// -----

func.func @v16i32_srs_v16i64(%arg0 : vector<16xi64>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<16xi64>, i32, vector<16xi32>
  %1 = aievec.srs %arg0, %c5 : vector<16xi64>, i32, vector<16xi32>
  return
}

// CHECK-LABEL: @v16i32_srs_v16i64
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi64>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.I512.v16.acc64.srs"(
// CHECK-SAME: [[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<16xi32>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.I512.v16.acc64.srs"(
// CHECK-SAME: [[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<16xi32>

// -----

func.func @v16i16_srs_v16i32(%arg0 : vector<16xi32>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<16xi32>, i32, vector<16xi16>
  %1 = aievec.srs %arg0, %c5 : vector<16xi32>, i32, vector<16xi16>
  return
}

// CHECK-LABEL: @v16i16_srs_v16i32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xi32> to vector<8xi64>
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.I256.v16.acc32.srs"(
// CHECK-SAME: [[BITCAST0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<8xi64>, i32, i32) -> vector<16xi16>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<16xi32> to vector<8xi64>
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.I256.v16.acc32.srs"(
// CHECK-SAME: [[BITCAST1]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<8xi64>, i32, i32) -> vector<16xi16>

// -----

func.func @v32i8_srs_v32i32(%arg0 : vector<32xi32>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<32xi32>, i32, vector<32xi8>
  %1 = aievec.srs %arg0, %c5 : vector<32xi32>, i32, vector<32xi8>
  return
}

// CHECK-LABEL: @v32i8_srs_v32i32
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi32>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi32> to vector<16xi64>
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.I256.v32.acc32.srs"(
// CHECK-SAME: [[BITCAST0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<32xi8>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi32> to vector<16xi64>
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.I256.v32.acc32.srs"(
// CHECK-SAME: [[BITCAST1]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<32xi8>

// -----

func.func @v16i16_srs_v16i64(%arg0 : vector<16xi64>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<16xi64>, i32, vector<16xi16>
  %1 = aievec.srs %arg0, %c5 : vector<16xi64>, i32, vector<16xi16>
  return
}

// CHECK-LABEL: @v16i16_srs_v16i64
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi64>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.I256.v16.acc64.srs"(
// CHECK-SAME: [[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<16xi16>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.I256.v16.acc64.srs"(
// CHECK-SAME: [[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<16xi64>, i32, i32) -> vector<16xi16>

// -----

func.func @v8i32_srs_v8i64(%arg0 : vector<8xi64>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<8xi64>, i32, vector<8xi32>
  %1 = aievec.srs %arg0, %c5 : vector<8xi64>, i32, vector<8xi32>
  return
}

// CHECK-LABEL: @v8i32_srs_v8i64
// CHECK-SAME: %[[ARG0:.*]]: vector<8xi64>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.I256.v8.acc64.srs"(
// CHECK-SAME: [[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<8xi64>, i32, i32) -> vector<8xi32>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.I256.v8.acc64.srs"(
// CHECK-SAME: [[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<8xi64>, i32, i32) -> vector<8xi32>

// -----

func.func @v16bf16_srs_v16f32(%arg0 : vector<16xf32>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<16xf32>, i32, vector<16xbf16>
  %1 = aievec.srs %arg0, %c5 : vector<16xf32>, i32, vector<16xbf16>
  return
}

// CHECK-LABEL: @v16bf16_srs_v16f32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf32>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<8xi64>
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(
// CHECK-SAME: [[BITCAST0]]) : 
// CHECK-SAME: (vector<8xi64>) -> vector<16xbf16>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<8xi64>
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(
// CHECK-SAME: [[BITCAST1]]) : 
// CHECK-SAME: (vector<8xi64>) -> vector<16xbf16>

// -----

func.func @v32bf16_srs_v32f32(%arg0 : vector<32xf32>) {
  %c0 = arith.constant 0 : i32
  %0 = aievec.srs %arg0, %c0 : vector<32xf32>, i32, vector<32xbf16>
  return
}

// CHECK-LABEL: @v32bf16_srs_v32f32
// CHECK-SAME: %[[ARG0:.*]]: vector<32xf32>
// CHECK: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[INDEX0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[INDEX1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST0]], %[[INDEX0]]) :
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[EXT0]] : vector<16xi32> to vector<8xi64>
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(
// CHECK-SAME: %[[BITCAST1]]) :
// CHECK-SAME: (vector<8xi64>) -> vector<16xbf16>
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG0]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I512.I1024"(
// CHECK-SAME: %[[BITCAST2]], %[[INDEX1]]) :
// CHECK-SAME: (vector<32xi32>, i32) -> vector<16xi32>
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[EXT1]] : vector<16xi32> to vector<8xi64>
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.v16accfloat.to.v16bf16"(
// CHECK-SAME: %[[BITCAST3]]) :
// CHECK-SAME: (vector<8xi64>) -> vector<16xbf16>
// CHECK-NEXT: %[[BITCAST4:.*]] = llvm.bitcast %[[SRS0]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[BITCAST5:.*]] = llvm.bitcast %[[SRS1]] : vector<16xbf16> to vector<8xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I512.I256"(
// CHECK-SAME: %[[BITCAST4]], %[[BITCAST5]]) :
// CHECK-SAME: (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
// CHECK-NEXT: %[[BITCAST6:.*]] = llvm.bitcast %[[CONCAT]] : vector<16xi32> to vector<32xbf16>
