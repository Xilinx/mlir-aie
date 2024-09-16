// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm | FileCheck %s

func.func @v16i32_ups_v16i16(%arg0 : vector<16xi16>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<16xi16>, vector<16xi32>
  %1 = aievec.ups %arg0 {shift = 5 : i8} : vector<16xi16>, vector<16xi32>
  return 
}

// CHECK-LABEL: @v16i32_ups_v16i16
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi16>
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.acc32.v16.I256.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<16xi16>, i32, i32) -> vector<8xi64>
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[SRS0]] : vector<8xi64> to vector<16xi32>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.acc32.v16.I256.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<16xi16>, i32, i32) -> vector<8xi64>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[SRS1]] : vector<8xi64> to vector<16xi32>

// -----

func.func @v8acc64_ups_v8i32(%arg0 : vector<8xi32>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<8xi32>, vector<8xi64>
  %1 = aievec.ups %arg0 {shift = 5 : i8} : vector<8xi32>, vector<8xi64>
  return 
}

// CHECK-LABEL: @v8acc64_ups_v8i32
// CHECK-SAME: %[[ARG0:.*]]: vector<8xi32>
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.acc64.v8.I256.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<8xi32>, i32, i32) -> vector<8xi64>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.acc64.v8.I256.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<8xi32>, i32, i32) -> vector<8xi64>

// -----

func.func @v32i32_ups_v32i16(%arg0 : vector<32xi16>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<32xi16>, vector<32xi32>
  %1 = aievec.ups %arg0 {shift = 5 : i8} : vector<32xi16>, vector<32xi32>
  return 
}

// CHECK-LABEL: @v32i32_ups_v32i16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.acc32.v32.I512.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<32xi16>, i32, i32) -> vector<16xi64>
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[SRS0]] : vector<16xi64> to vector<32xi32>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.acc32.v32.I512.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<32xi16>, i32, i32) -> vector<16xi64>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[SRS1]] : vector<16xi64> to vector<32xi32>

// -----

func.func @v16acc64_ups_v16i32(%arg0 : vector<16xi32>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
  %1 = aievec.ups %arg0 {shift = 5 : i8} : vector<16xi32>, vector<16xi64>
  return 
}

// CHECK-LABEL: @v16acc64_ups_v16i32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.acc64.v16.I512.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<16xi32>, i32, i32) -> vector<16xi64>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.acc64.v16.I512.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<16xi32>, i32, i32) -> vector<16xi64>

// -----

func.func @v16acc64_ups_v16i16(%arg0 : vector<16xi16>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<16xi16>, vector<16xi64>
  %1 = aievec.ups %arg0 {shift = 5 : i8} : vector<16xi16>, vector<16xi64>
  return 
}

// CHECK-LABEL: @v16acc64_ups_v16i16
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi16>
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.acc64.v16.I256.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<16xi16>, i32, i32) -> vector<16xi64>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.acc64.v16.I256.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<16xi16>, i32, i32) -> vector<16xi64>

// -----

func.func @v32i32_ups_v32i8(%arg0 : vector<32xi8>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<32xi8>, vector<32xi32>
  %1 = aievec.ups %arg0 {shift = 5 : i8} : vector<32xi8>, vector<32xi32>
  return 
}

// CHECK-LABEL: @v32i32_ups_v32i8
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi8>
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.acc32.v32.I256.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<32xi8>, i32, i32) -> vector<16xi64>
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[SRS0]] : vector<16xi64> to vector<32xi32>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.acc32.v32.I256.ups"(
// CHECK-SAME: [[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<32xi8>, i32, i32) -> vector<16xi64>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[SRS1]] : vector<16xi64> to vector<32xi32>

// -----

func.func @v16f32_ups_v16bf16(%arg0 : vector<16xbf16>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<16xbf16>, vector<16xf32> 
  %1 = aievec.ups %arg0 {shift = 5 : i8} : vector<16xbf16>, vector<16xf32> 
  return
}

// CHECK-LABEL: @v16f32_ups_v16bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2.v16bf16.to.v16accfloat"(
// CHECK-SAME: [[ARG0]]) : 
// CHECK-SAME: (vector<16xbf16>) -> vector<8xi64>
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[SRS0]] : vector<8xi64> to vector<16xf32> 
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2.v16bf16.to.v16accfloat"(
// CHECK-SAME: [[ARG0]]) : 
// CHECK-SAME: (vector<16xbf16>) -> vector<8xi64>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[SRS1]] : vector<8xi64> to vector<16xf32> 

// -----

func.func @v32f32_ups_v32bf16(%arg0 : vector<32xbf16>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<32xbf16>, vector<32xf32> 
  return
}

// CHECK-LABEL: @v32f32_ups_v32bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>
// CHECK: %[[INDEX0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[INDEX1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xbf16> to vector<16xi32>
// CHECK-NEXT: %[[EXT0:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST0]], %[[INDEX0]]) :
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[EXT0]] : vector<8xi32> to vector<16xbf16>
// CHECK-NEXT: %[[UPS0:.*]] = "xllvm.intr.aie2.v16bf16.to.v16accfloat"(
// CHECK-SAME: %[[BITCAST1]]) :
// CHECK-SAME: (vector<16xbf16>) -> vector<8xi64>
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG0]] : vector<32xbf16> to vector<16xi32>
// CHECK-NEXT: %[[EXT1:.*]] = "xllvm.intr.aie2.ext.I256.I512"(
// CHECK-SAME: %[[BITCAST2]], %[[INDEX1]]) :
// CHECK-SAME: (vector<16xi32>, i32) -> vector<8xi32>
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[EXT1]] : vector<8xi32> to vector<16xbf16>
// CHECK-NEXT: %[[UPS1:.*]] = "xllvm.intr.aie2.v16bf16.to.v16accfloat"(
// CHECK-SAME: %[[BITCAST3]]) :
// CHECK-SAME: (vector<16xbf16>) -> vector<8xi64>
// CHECK-NEXT: %[[BITCAST4:.*]] = llvm.bitcast %[[UPS0]] : vector<8xi64> to vector<16xi32>
// CHECK-NEXT: %[[BITCAST5:.*]] = llvm.bitcast %[[UPS1]] : vector<8xi64> to vector<16xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = "xllvm.intr.aie2.concat.I1024.I512"(
// CHECK-SAME: %[[BITCAST4]], %[[BITCAST5]]) :
// CHECK-SAME: (vector<16xi32>, vector<16xi32>) -> vector<32xi32>
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<32xf32>

// -----

// CHECK-LABEL: @multidim_ups_i8_to_i32
// CHECK-SAME: %[[V:.*]]: vector<4x8xi8>
// CHECK: %[[FV:.*]] = vector.shape_cast %[[V]]
// CHECK-SAME:                     : vector<4x8xi8> to vector<32xi8>
// CHECK: %[[FUPS:.*]] = "xllvm.intr.aie2.acc32.v32.I256.ups"(%[[FV]],
// CHECK-SAME:                             %{{[a-zA-Z0-9]+}}, %{{[a-zA-Z0-9]+}})
// CHECK-SAME:                     : (vector<32xi8>, i32, i32) -> vector<16xi64>
// CHECK: %[[FR:.*]] = llvm.bitcast %3 : vector<16xi64> to vector<32xi32>
// CHECK: %[[UPS:.*]] = vector.shape_cast %[[FR]]
// CHECK-sAME:                     : vector<32xi32> to vector<4x8xi32>
func.func @multidim_ups_i8_to_i32(%arg0 : vector<4x8xi8>) -> vector<4x8xi32> {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<4x8xi8>, vector<4x8xi32>
  return %0 : vector<4x8xi32>
}
