// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
module {
  func.func @test(%arg0: vector<32xi16>) {
    // check the parameters that go into separate constants
    %0 = aievec.select %arg0 {select = "0x00000000", xoffsets = "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", yoffsets = "0x00000000", yoffsets_hi = "0x00000000", ysquare = "0x0000", ystart = "0"} : vector<32xi16>, vector<32xi16>
    %1 = aievec.select %arg0 {select = "0xfedcba98", xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x0000", xstart = "2", yoffsets = "0x03020100", yoffsets_hi = "0x07060504", ysquare = "0x0000", ystart = "7"} : vector<32xi16>, vector<32xi16>

    // check the various combinations that make up the configuration value
    %2 = aievec.select %arg0 {select = "0x00000000", xoffsets = "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x3210", xstart = "0", yoffsets = "0x00000000", yoffsets_hi = "0x00000000", ysquare = "0x0000", ystart = "0"} : vector<32xi16>, vector<32xi16>
    %3 = aievec.select %arg0 {select = "0x00000000", xoffsets = "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", yoffsets = "0x00000000", yoffsets_hi = "0x00000000", ysquare = "0x3210", ystart = "0"} : vector<32xi16>, vector<32xi16>
    // all of the configuration register values
    %4 = aievec.select %arg0 {select = "0x00000000", xoffsets = "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x3210", xstart = "0", yoffsets = "0x00000000", yoffsets_hi = "0x00000000", ysquare = "0x3210", ystart = "0"} : vector<32xi16>, vector<32xi16>
    return
  }
}
// CHECK:      %0 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %1 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %2 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %3 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %4 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %5 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %6 = llvm.call @llvm.aie.prim.v32int16.select(%arg0, %0, %1, %2, %3, %4, %5) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>
// CHECK:      %7 = llvm.mlir.constant(-19088744 : i32) : i32
// CHECK-NEXT: %8 = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %9 = llvm.mlir.constant(7 : i32) : i32
// CHECK-NEXT: %10 = llvm.mlir.constant(dense<[50462976, 117835012]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %11 = llvm.mlir.constant(dense<[50462976, 117835012]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %12 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %13 = llvm.call @llvm.aie.prim.v32int16.select(%arg0, %7, %8, %9, %10, %11, %12) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>
// CHECK:      %14 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %15 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %16 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %17 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %18 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %19 = llvm.mlir.constant(dense<[0, 228]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %20 = llvm.call @llvm.aie.prim.v32int16.select(%arg0, %14, %15, %16, %17, %18, %19) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>
// CHECK:      %21 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %22 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %23 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %24 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %25 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %26 = llvm.mlir.constant(dense<[0, 478150656]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %27 = llvm.call @llvm.aie.prim.v32int16.select(%arg0, %21, %22, %23, %24, %25, %26) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>
// CHECK:      %28 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %29 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %30 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %31 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %32 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %33 = llvm.mlir.constant(dense<[0, 478150884]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %34 = llvm.call @llvm.aie.prim.v32int16.select(%arg0, %28, %29, %30, %31, %32, %33) : (vector<32xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<32xi16>
