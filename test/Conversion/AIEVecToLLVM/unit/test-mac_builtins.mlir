// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
module {
  func.func @test() {
    %v16i48 = llvm.mlir.undef : vector<16xi48>
    %v32i16 = llvm.mlir.undef: vector<32xi16>
    %v16i16 = llvm.mlir.undef: vector<16xi16>
    %v64i8 = llvm.mlir.undef: vector<64xi8>
    %v32i8 = llvm.mlir.undef: vector<32xi8>
    %v8i48 = llvm.mlir.undef: vector<8xi48>
    %v16i32 = llvm.mlir.undef: vector<16xi32>
    %v8i32 = llvm.mlir.undef: vector<8xi32>
    %v8i80 = llvm.mlir.undef: vector<8xi80>
    %0 = aievec.mac %v32i16, %v16i16, %v16i48 : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %1 = aievec.mac %v64i8, %v32i8, %v8i48 : vector<64xi8>, vector<32xi8>, vector<8xi48>
    %2 = aievec.mac %v16i32, %v8i32, %v8i80 : vector<16xi32>, vector<8xi32>, vector<8xi80>
    return
  }
}

// The function declarations are sorted
// CHECK: llvm.func @__builtin_aie_mac8_v16int32_v8int32_bm_sw80(vector<16xi32>, vector<8xi32>, vector<8xi80>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<8xi80>
// CHECK: llvm.func @__builtin_aie_mac8_v64int8_v32int8_am_sw48(vector<64xi8>, vector<32xi8>, vector<8xi48>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<8xi48>
// CHECK: llvm.func @__builtin_aie_mac16_v32int16_v16int16_bm_sw48(vector<32xi16>, vector<16xi16>, vector<16xi48>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
