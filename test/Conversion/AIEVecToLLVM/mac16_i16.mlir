// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
module {
  func.func @mac16(%arg0: vector<32xi16>, %arg1: vector<16xi16>, %arg2: vector<16xi48>) {
    %0 = aievec_aie1.mac %arg0, %arg1, %arg2 {xoffsets= "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "1", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    return
  }
}
// CHECK: llvm.func @llvm.aie.mac16.v32int16(vector<32xi16>, vector<16xi16>, vector<16xi48>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
// CHECK: [[XSTART:%[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: [[YSTART:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ZSTART:%[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: [[XOFFS:%[0-9]+]] = llvm.mlir.constant(dense<[50462976, 117835012]> : vector<2xi32>) : vector<2xi32>
// CHECK: [[ZOFFS:%[0-9]+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%[0-9]+]] = llvm.mlir.constant(dense<[256, 148]> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.mac16.v32int16(%arg0, %arg1, %arg2, [[XSTART]], [[YSTART]], [[ZSTART]], [[XOFFS]], [[ZOFFS]], [[CONF]]) : (vector<32xi16>, vector<16xi16>, vector<16xi48>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
