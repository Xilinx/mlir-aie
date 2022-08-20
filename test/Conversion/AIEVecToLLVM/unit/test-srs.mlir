// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
module {
  func.func @test() {
    %v8i48 = llvm.mlir.undef: vector<8xi48>
    %v16i48 = llvm.mlir.undef: vector<16xi48>
    %v4i80 = llvm.mlir.undef: vector<4xi80>
    %v8i80 = llvm.mlir.undef: vector<8xi80>
    // sweep the variants of SRS and values of shift
    %0 = aievec.srs %v16i48 {shift = 0 : i8} : vector<16xi48>, vector<16xi8>
    %1 = aievec.srs %v16i48 {shift = 1 : i8} : vector<16xi48>, vector<16xui8>
    %2 = aievec.srs %v16i48 {shift = 2 : i8} : vector<16xi48>, vector<16xi16>
    %3 = aievec.srs %v8i48 {shift = 3 : i8} : vector<8xi48>, vector<8xi32>
    %4 = aievec.srs %v4i80 {shift = 4 : i8} : vector<4xi80>, vector<4xi32>
    return
  }
}

// The function declarations are in reverse order of their declarations
// CHECK:      llvm.func @llvm.aie.srs.v4i32(vector<4xi80>, i8) -> vector<4xi32>
// CHECK:      llvm.func @llvm.aie.lsrs.v8i32(vector<8xi48>, i8) -> vector<8xi32>
// CHECK:      llvm.func @llvm.aie.srs.v16i16(vector<16xi48>, i8) -> vector<16xi16>
// CHECK:      llvm.func @llvm.aie.usrs.v16i8(vector<16xi48>, i8) -> vector<16xui8>
// CHECK:      llvm.func @llvm.aie.bsrs.v16i8(vector<16xi48>, i8) -> vector<16xi8>
// CHECK:      %0 = llvm.mlir.undef : vector<8xi48>
// CHECK-NEXT: %1 = llvm.mlir.undef : vector<16xi48>
// CHECK-NEXT: %2 = llvm.mlir.undef : vector<4xi80>
// CHECK-NEXT: %3 = llvm.mlir.undef : vector<8xi80>
// CHECK-NEXT: %4 = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT: %5 = llvm.call @llvm.aie.bsrs.v16i8(%1, %4) : (vector<16xi48>, i8) -> vector<16xi8>
// CHECK-NEXT: %6 = llvm.mlir.constant(1 : i8) : i8
// CHECK-NEXT: %7 = llvm.call @llvm.aie.usrs.v16i8(%1, %6) : (vector<16xi48>, i8) -> vector<16xui8>
// CHECK-NEXT: %8 = llvm.mlir.constant(2 : i8) : i8
// CHECK-NEXT: %9 = llvm.call @llvm.aie.srs.v16i16(%1, %8) : (vector<16xi48>, i8) -> vector<16xi16>
// CHECK-NEXT: %10 = llvm.mlir.constant(3 : i8) : i8
// CHECK-NEXT: %11 = llvm.call @llvm.aie.lsrs.v8i32(%0, %10) : (vector<8xi48>, i8) -> vector<8xi32>
// CHECK-NEXT: %12 = llvm.mlir.constant(4 : i8) : i8
// CHECK-NEXT: %13 = llvm.call @llvm.aie.srs.v4i32(%2, %12) : (vector<4xi80>, i8) -> vector<4xi32>
