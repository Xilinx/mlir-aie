// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
module {
  func.func @test(%arg0: vector<32xi16>, %arg1: vector<16xi16>) {
    // check the parameters that go into separate constants
    %0 = aievec_aie1.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", xstep = "0", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x0000", zstart = "0", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %1 = aievec_aie1.mul %arg0, %arg1 {xoffsets= "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x0000", xstart = "2", xstep = "0", zoffsets = "0x03020100", zoffsets_hi = "0x07060504", zsquare = "0x0000", zstart = "7", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>

    // check the various combinations that make up the configuration value
    %2 = aievec_aie1.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x3210", xstart = "0", xstep = "0", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x0000", zstart = "0", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %3 = aievec_aie1.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", xstep = "0", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x3210", zstart = "0", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %4 = aievec_aie1.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", xstep = "4", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x0000", zstart = "0", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %5 = aievec_aie1.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", xstep = "0", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x0000", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    // all of the configuration register values
    %6 = aievec_aie1.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x3210", xstart = "0", xstep = "4", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x3210", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    return
  }
}
// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ZSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[ZOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, [[XSTART]], [[YSTART]], [[ZSTART]], [[XOFFS]], [[ZOFFS]], [[CONF]]) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>

// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ZSTART:%.+]] = llvm.mlir.constant(7 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<[50462976, 117835012]> : vector<2xi32>) : vector<2xi32>
// CHECK: [[ZOFFS:%.+]] = llvm.mlir.constant(dense<[50462976, 117835012]> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, [[XSTART]], [[YSTART]], [[ZSTART]], [[XOFFS]], [[ZOFFS]], [[CONF]]) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>

// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ZSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[ZOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<[0, 228]> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, [[XSTART]], [[YSTART]], [[ZSTART]], [[XOFFS]], [[ZOFFS]], [[CONF]]) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>

// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ZSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[ZOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<[0, 58368]> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, [[XSTART]], [[YSTART]], [[ZSTART]], [[XOFFS]], [[ZOFFS]], [[CONF]]) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>

// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ZSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[ZOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<[4, 0]> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, [[XSTART]], [[YSTART]], [[ZSTART]], [[XOFFS]], [[ZOFFS]], [[CONF]]) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>

// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ZSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[ZOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<[256, 0]> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, [[XSTART]], [[YSTART]], [[ZSTART]], [[XOFFS]], [[ZOFFS]], [[CONF]]) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>

// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[ZSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[ZOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<[260, 58596]> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, [[XSTART]], [[YSTART]], [[ZSTART]], [[XOFFS]], [[ZOFFS]], [[CONF]]) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
