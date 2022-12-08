// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
module {
  func.func @test() {
    %0 = llvm.mlir.undef: vector<16xi16>
    %1 = aievec.pack %0 : vector<16xi16>, vector<16xi8>
    return
  }
}
// CHECK: %1 = llvm.call @llvm.aie.pack.v16int16(%0) : (vector<16xi16>) -> vector<16xi8>
