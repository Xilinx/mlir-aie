// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
// Test loads and updates to a vector register
module {
  func.func @test(%arg0: memref<4x32x64xi16>) {
    %i = arith.constant 1 : index
    %j = arith.constant 2 : index
    %k = arith.constant 3 : index
    %0 = aievec.upd %arg0[%i, %j, %k] {index = 0 : i8, offset = 0: si32} : memref<4x32x64xi16>, vector<32xi16>
    %1 = aievec.upd %arg0[%i, %j, %k], %0 {index = 1 : i8, offset = 0: si32} : memref<4x32x64xi16>, vector<32xi16>
    return
  }
}
// CHECK: %4 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i16>, ptr<i16>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK: %5 = llvm.mlir.constant(2048 : index) : i64
// CHECK: %6 = llvm.mul %1, %5 : i64
// CHECK: %7 = llvm.mlir.constant(64 : index) : i64
// CHECK: %8 = llvm.mul %2, %7 : i64
// CHECK: %9 = llvm.add %6, %8 : i64
// CHECK: %10 = llvm.add %9, %3 : i64
// CHECK: %11 = llvm.getelementptr %4[%10] : (!llvm.ptr<i16>, i64) -> !llvm.ptr<i16>
// CHECK: %12 = llvm.bitcast %11 : !llvm.ptr<i16> to !llvm.ptr<vector<16xi16>>
// CHECK: %13 = llvm.load %12 {alignment = 1 : i64} : !llvm.ptr<vector<16xi16>>
// CHECK: %14 = llvm.call @llvm.aie.v32int16.undef() : () -> vector<32xi16>
// CHECK: %15 = llvm.call @llvm.aie.upd.w.v32int16.lo(%14, %13) : (vector<32xi16>, vector<16xi16>) -> vector<32xi16>
// CHECK: %16 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i16>, ptr<i16>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK: %17 = llvm.mlir.constant(2048 : index) : i64
// CHECK: %18 = llvm.mul %1, %17 : i64
// CHECK: %19 = llvm.mlir.constant(64 : index) : i64
// CHECK: %20 = llvm.mul %2, %19 : i64
// CHECK: %21 = llvm.add %18, %20 : i64
// CHECK: %22 = llvm.add %21, %3 : i64
// CHECK: %23 = llvm.getelementptr %16[%22] : (!llvm.ptr<i16>, i64) -> !llvm.ptr<i16>
// CHECK: %24 = llvm.bitcast %23 : !llvm.ptr<i16> to !llvm.ptr<vector<16xi16>>
// CHECK: %25 = llvm.load %24 {alignment = 1 : i64} : !llvm.ptr<vector<16xi16>>
// CHECK: %26 = llvm.call @llvm.aie.upd.w.v32int16.hi(%15, %25) : (vector<32xi16>, vector<16xi16>) -> vector<32xi16>
