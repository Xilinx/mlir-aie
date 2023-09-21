// RUN: aie-opt %s --convert-aievec-to-llvm -canonicalize | FileCheck %s
module {
  func.func @test() {
    %v16i8 = llvm.mlir.undef : vector<16xi8>
    %v16ui8 = llvm.mlir.undef : vector<16xui8>
    %v16i16 = llvm.mlir.undef : vector<16xi16>
    %v8i32 = llvm.mlir.undef : vector<8xi32>
    %v4i32 = llvm.mlir.undef : vector<4xi32>
    %v8i64 = llvm.mlir.undef : vector<8xi64>
    // sweep the variants of UPS and values of shift
    %0 = aievec.ups %v16i8 {shift = 0 : i8} : vector<16xi8>, vector<16xi48>
    %1 = aievec.ups %v16ui8 {shift = 1 : i8} : vector<16xui8>, vector<16xi48>
    %2 = aievec.ups %v16i16 {shift = 2 : i8} : vector<16xi16>, vector<16xi48>
    %3 = aievec.ups %v8i32 {shift = 3 : i8} : vector<8xi32>, vector<8xi80>
    %4 = aievec.ups %v4i32 {shift = 4 : i8} : vector<4xi32>, vector<4xi80>
    %5 = aievec.ups %v8i64 {shift = 5 : i8} : vector<8xi64>, vector<8xi80>
    return
  }
}

// CHECK-DAG: llvm.func @llvm.aie.bups.v16acc48.v16i8(vector<16xi8>, i32) -> vector<16xi48>
// CHECK-DAG: llvm.func @llvm.aie.uups.v16acc48.v16i8(vector<16xui8>, i32) -> vector<16xi48>
// CHECK-DAG: llvm.func @llvm.aie.ups.v16acc48.v16i16(vector<16xi16>, i32) -> vector<16xi48>
// CHECK-DAG: llvm.func @llvm.aie.ups.v8acc80.v8i32(vector<8xi32>, i32) -> vector<8xi80>
// CHECK-DAG: llvm.func @llvm.aie.ups.v4acc80.v4i32(vector<4xi32>, i32) -> vector<4xi80>
// CHECK-DAG: llvm.func @llvm.aie.lups.v8acc80.v8i64(vector<8xi64>, i32) -> vector<8xi80>
// CHECK-LABEL: func.func @test() {
// CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-DAG: %[[C3:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-DAG: %[[C5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-DAG: %[[V16xI8:.*]] = llvm.mlir.undef : vector<16xi8>
// CHECK-DAG: %[[V16xUI8:.*]] = llvm.mlir.undef : vector<16xui8>
// CHECK-DAG: %[[V16xI16:.*]] = llvm.mlir.undef : vector<16xi16>
// CHECK-DAG: %[[V8xI32:.*]] = llvm.mlir.undef : vector<8xi32>
// CHECK-DAG: %[[V4xI32:.*]] = llvm.mlir.undef : vector<4xi32>
// CHECK-DAG: %[[V8xI64:.*]] = llvm.mlir.undef : vector<8xi64>
// CHECK:     llvm.call @llvm.aie.bups.v16acc48.v16i8(%[[V16xI8]], %[[C0]]) : (vector<16xi8>, i32) -> vector<16xi48>
// CHECK:     llvm.call @llvm.aie.uups.v16acc48.v16i8(%[[V16xUI8]], %[[C1]]) : (vector<16xui8>, i32) -> vector<16xi48>
// CHECK:     llvm.call @llvm.aie.ups.v16acc48.v16i16(%[[V16xI16]], %[[C2]]) : (vector<16xi16>, i32) -> vector<16xi48>
// CHECK:     llvm.call @llvm.aie.ups.v8acc80.v8i32(%[[V8xI32]], %[[C3]]) : (vector<8xi32>, i32) -> vector<8xi80>
// CHECK:     llvm.call @llvm.aie.ups.v4acc80.v4i32(%[[V4xI32]], %[[C4]]) : (vector<4xi32>, i32) -> vector<4xi80>
// CHECK:     llvm.call @llvm.aie.lups.v8acc80.v8i64(%[[V8xI64]], %[[C5]]) : (vector<8xi64>, i32) -> vector<8xi80>
