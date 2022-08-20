// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
module {
  func.func @test(%arg0: vector<32xi16>, %arg1: vector<16xi16>) {
    // check the parameters that go into separate constants
    %0 = aievec.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", xstep = "0", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x0000", zstart = "0", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %1 = aievec.mul %arg0, %arg1 {xoffsets= "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x0000", xstart = "2", xstep = "0", zoffsets = "0x03020100", zoffsets_hi = "0x07060504", zsquare = "0x0000", zstart = "7", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>

    // check the various combinations that make up the configuration value
    %2 = aievec.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x3210", xstart = "0", xstep = "0", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x0000", zstart = "0", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %3 = aievec.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", xstep = "0", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x3210", zstart = "0", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %4 = aievec.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", xstep = "4", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x0000", zstart = "0", zstep = "0"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    %5 = aievec.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x0000", xstart = "0", xstep = "0", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x0000", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    // all of the configuration register values
    %6 = aievec.mul %arg0, %arg1 {xoffsets= "0x00000000", xoffsets_hi = "0x00000000", xsquare = "0x3210", xstart = "0", xstep = "4", zoffsets = "0x00000000", zoffsets_hi = "0x00000000", zsquare = "0x3210", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
    return
  }
}
// CHECK:      %0 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %1 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %2 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %3 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %4 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %5 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %6 = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, %0, %1, %2, %3, %4, %5) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
// CHECK:      %7 = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %8 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %9 = llvm.mlir.constant(7 : i32) : i32
// CHECK-NEXT: %10 = llvm.mlir.constant(dense<[50462976, 117835012]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %11 = llvm.mlir.constant(dense<[50462976, 117835012]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %12 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %13 = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, %7, %8, %9, %10, %11, %12) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
// CHECK:      %14 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %15 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %16 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %17 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %18 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %19 = llvm.mlir.constant(dense<[0, 228]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %20 = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, %14, %15, %16, %17, %18, %19) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
// CHECK:      %21 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %22 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %23 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %24 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %25 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %26 = llvm.mlir.constant(dense<[0, 58368]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %27 = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, %21, %22, %23, %24, %25, %26) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
// CHECK:      %28 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %29 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %30 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %31 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %32 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %33 = llvm.mlir.constant(dense<[4, 0]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %34 = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, %28, %29, %30, %31, %32, %33) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
// CHECK:      %35 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %36 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %37 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %38 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %39 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %40 = llvm.mlir.constant(dense<[256, 0]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %41 = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, %35, %36, %37, %38, %39, %40) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
// CHECK:      %42 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %43 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %44 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %45 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %46 = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %47 = llvm.mlir.constant(dense<[260, 58596]> : vector<2xi32>) : vector<2xi32>
// CHECK-NEXT: %48 = llvm.call @llvm.aie.mul16.v32int16(%arg0, %arg1, %42, %43, %44, %45, %46, %47) : (vector<32xi16>, vector<16xi16>, i32, i32, i32, vector<2xi32>, vector<2xi32>, vector<2xi32>) -> vector<16xi48>
