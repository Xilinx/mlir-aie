// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
module {
  func.func @test(%arg0: vector<32xi16>) {
    // check the parameters that go into separate constants
    %0 = aievec_aie1.select %arg0 {select = "0x00000000", xoffsets = "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", yoffsets = "0x00000000", yoffsets_hi = "0x00000000", ysquare = "0x0000", ystart = "0"} : vector<32xi16>, vector<32xi16>
    %1 = aievec_aie1.select %arg0 {select = "0xfedcba98", xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x0000", xstart = "2", yoffsets = "0x03020100", yoffsets_hi = "0x07060504", ysquare = "0x0000", ystart = "7"} : vector<32xi16>, vector<32xi16>

    // check the various combinations that make up the configuration value
    %2 = aievec_aie1.select %arg0 {select = "0x00000000", xoffsets = "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x3210", xstart = "0", yoffsets = "0x00000000", yoffsets_hi = "0x00000000", ysquare = "0x0000", ystart = "0"} : vector<32xi16>, vector<32xi16>
    %3 = aievec_aie1.select %arg0 {select = "0x00000000", xoffsets = "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", yoffsets = "0x00000000", yoffsets_hi = "0x00000000", ysquare = "0x3210", ystart = "0"} : vector<32xi16>, vector<32xi16>
    // all of the configuration register values
    %4 = aievec_aie1.select %arg0 {select = "0x00000000", xoffsets = "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x3210", xstart = "0", yoffsets = "0x00000000", yoffsets_hi = "0x00000000", ysquare = "0x3210", ystart = "0"} : vector<32xi16>, vector<32xi16>
    return
  }
}
// CHECK: [[SELECT:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[YOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.prim.v32int16.select(%arg0, [[SELECT]], [[XSTART]], [[YSTART]], [[XOFFS]], [[YOFFS]], [[CONF]]) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>

// CHECK: [[SELECT:%.+]] = llvm.mlir.constant(-19088744 : i32) : i32
// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(7 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<[50462976, 117835012]> : vector<2xi32>) : vector<2xi32>
// CHECK: [[YOFFS:%.+]] = llvm.mlir.constant(dense<[50462976, 117835012]> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.prim.v32int16.select(%arg0, [[SELECT]], [[XSTART]], [[YSTART]], [[XOFFS]], [[YOFFS]], [[CONF]]) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>

// CHECK: [[SELECT:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[YOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<[0, 228]> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.prim.v32int16.select(%arg0, [[SELECT]], [[XSTART]], [[YSTART]], [[XOFFS]], [[YOFFS]], [[CONF]]) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>

// CHECK: [[SELECT:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[YOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<[0, 478150656]> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.prim.v32int16.select(%arg0, [[SELECT]], [[XSTART]], [[YSTART]], [[XOFFS]], [[YOFFS]], [[CONF]]) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>

// CHECK: [[SELECT:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[YSTART:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[XOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[YOFFS:%.+]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK: [[CONF:%.+]] = llvm.mlir.constant(dense<[0, 478150884]> : vector<2xi32>) : vector<2xi32>
// CHECK: {{.*}} = llvm.call @llvm.aie.prim.v32int16.select(%arg0, [[SELECT]], [[XSTART]], [[YSTART]], [[XOFFS]], [[YOFFS]], [[CONF]]) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>
