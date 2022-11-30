// RUN: aie-opt %s --lower-aievec | FileCheck %s
// Test a direct load to vector register that does not actually need an update
module {
  func.func @test(%arg0: memref<4x32x64xi16>) {
    %i = arith.constant 1 : index
    %j = arith.constant 2 : index
    %k = arith.constant 3 : index
    %0 = aievec.upd %arg0[%i, %j, %k] {index = 0 : i8, offset = 0: si32} : memref<4x32x64xi16>, vector<16xi16>
    return
  }
}
// CHECK:      %4 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i16>, ptr<i16>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %5 = llvm.mlir.constant(2048 : index) : i64
// CHECK-NEXT: %6 = llvm.mul %1, %5 : i64
// CHECK-NEXT: %7 = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: %8 = llvm.mul %2, %7 : i64
// CHECK-NEXT: %9 = llvm.add %6, %8 : i64
// CHECK-NEXT: %10 = llvm.add %9, %3 : i64
// CHECK-NEXT: %11 = llvm.getelementptr %4[%10] : (!llvm.ptr<i16>, i64) -> !llvm.ptr<i16>
// CHECK-NEXT: %12 = llvm.bitcast %11 : !llvm.ptr<i16> to !llvm.ptr<vector<16xi16>>
// CHECK-NEXT: %13 = llvm.load %12 {alignment = 1 : i64} : !llvm.ptr<vector<16xi16>>
