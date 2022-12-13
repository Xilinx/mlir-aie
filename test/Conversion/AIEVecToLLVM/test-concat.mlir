// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
// Test a direct load to vector register that does not actually need an update
module {
  func.func @test() {
    %0 = llvm.mlir.undef : vector<16xi16>
    %1 = llvm.mlir.undef : vector<16xi16>
    %2 = aievec.concat %0, %1 : vector<16xi16>, vector<32xi16>
    %3 = llvm.mlir.undef : vector<32xi8>
    %4 = llvm.mlir.undef : vector<32xi8>
    %5 = aievec.concat %3, %4 : vector<32xi8>, vector<64xi8>
    %6 = llvm.mlir.undef : vector<8xi32>
    %7 = llvm.mlir.undef : vector<8xi32>
    %8 = aievec.concat %6, %7 : vector<8xi32>, vector<16xi32>
    return
  }
}
// CHECK: [[UNDEF_V0:%.+]] = llvm.mlir.undef : vector<16xi16>
// CHECK: [[UNDEF_V1:%.+]] = llvm.mlir.undef : vector<16xi16>
// CHECK: {{.*}} = llvm.call @llvm.aie.concat.v16i16([[UNDEF_V0]], [[UNDEF_V1]]) : (vector<16xi16>, vector<16xi16>) -> vector<32xi16>
// CHECK: [[UNDEF_V0:%.+]] = llvm.mlir.undef : vector<32xi8>
// CHECK: [[UNDEF_V1:%.+]] = llvm.mlir.undef : vector<32xi8>
// CHECK: {{.*}} = llvm.call @llvm.aie.concat.v32i8([[UNDEF_V0]], [[UNDEF_V1]]) : (vector<32xi8>, vector<32xi8>) -> vector<64xi8>
// CHECK: [[UNDEF_V0:%.+]] = llvm.mlir.undef : vector<8xi32>
// CHECK: [[UNDEF_V1:%.+]] = llvm.mlir.undef : vector<8xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.concat.v8i32([[UNDEF_V0]], [[UNDEF_V1]]) : (vector<8xi32>, vector<8xi32>) -> vector<16xi32>
