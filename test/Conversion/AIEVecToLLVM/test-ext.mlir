// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
// Test a direct load to vector register that does not actually need an update
module {
  func.func @test() {
    %0 = llvm.mlir.undef : vector<32xi8>
    %1 = aievec.ext %0 {index = 0 : i8} : vector<32xi8>, vector<16xi8>
    %2 = aievec.ext %0 {index = 1 : i8} : vector<32xi8>, vector<16xi8>
    %3 = llvm.mlir.undef : vector<32xi16>
    %4 = aievec.ext %3 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
    %5 = aievec.ext %3 {index = 1 : i8} : vector<32xi16>, vector<16xi16>
    %6 = llvm.mlir.undef : vector<32xi32>
    %7 = aievec.ext %6 {index = 0 : i8} : vector<32xi32>, vector<16xi32>
    %8 = aievec.ext %6 {index = 1 : i8} : vector<32xi32>, vector<16xi32>
    return
  }
}
// CHECK: [[UNDEF_V32I8:%.+]] = llvm.mlir.undef : vector<32xi8>
// CHECK: {{.*}} = llvm.call @llvm.aie.ext.v.v32i8.lo([[UNDEF_V32I8]]) : (vector<32xi8>) -> vector<16xi8>
// CHECK: {{.*}} = llvm.call @llvm.aie.ext.v.v32i8.hi([[UNDEF_V32I8]]) : (vector<32xi8>) -> vector<16xi8>
// CHECK: [[UNDEF_V32I16:%.+]] = llvm.mlir.undef : vector<32xi16>
// CHECK: {{.*}} = llvm.call @llvm.aie.ext.w.v32i16.lo([[UNDEF_V32I16]]) : (vector<32xi16>) -> vector<16xi16>
// CHECK: {{.*}} = llvm.call @llvm.aie.ext.w.v32i16.hi([[UNDEF_V32I16]]) : (vector<32xi16>) -> vector<16xi16>
// CHECK: [[UNDEF_V32I32:%.+]] = llvm.mlir.undef : vector<32xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.ext.x.v32i32.lo([[UNDEF_V32I32]]) : (vector<32xi32>) -> vector<16xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.ext.x.v32i32.hi([[UNDEF_V32I32]]) : (vector<32xi32>) -> vector<16xi32>
