// RUN: aie-opt %s --convert-linalg-to-affine-loops --affine-loop-unroll="unroll-full unroll-full-threshold=3" --canonicalize -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize -unaligned-loads-check=false -split-input-file | FileCheck %s

//The affine dialect code is generated from the following linalg operator, with the maps inlined. 
//CHECK-LABEL: func.func @conv_2d(%arg0: memref<10x3x256x256xf32>, %arg1: memref<10x3x3x3xf32>, %arg2: memref<10x10x254x254xf32>) {
func.func @conv_2d(%input: memref<10x3x256x256xf32>, %filter: memref<10x3x3x3xf32>, %output: memref<10x10x254x254xf32>) {
  linalg.conv_2d_nchw_fchw{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<10x3x256x256xf32>, memref<10x3x3x3xf32>)
    outs (%output: memref<10x10x254x254xf32>)
  return
}

//CHECK-NEXT: %c2 = arith.constant 2 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %c0_0 = arith.constant 0 : index
//CHECK-NEXT: %c10 = arith.constant 10 : index
//CHECK-NEXT: %c1_1 = arith.constant 1 : index
//CHECK-NEXT: scf.for %arg3 = %c0_0 to %c10 step %c1_1 {
//CHECK-NEXT:   %c0_2 = arith.constant 0 : index
//CHECK-NEXT:   %c10_3 = arith.constant 10 : index
//CHECK-NEXT:   %c1_4 = arith.constant 1 : index
//CHECK-NEXT:   scf.for %arg4 = %c0_2 to %c10_3 step %c1_4 {
//CHECK-NEXT:     %0 = aievec.upd %arg1[%arg4, %c0, %c0, %c0] {index = 0 : i8, offset = 0 : i32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %1 = aievec.upd %arg1[%arg4, %c0, %c2, %c2] {index = 0 : i8, offset = 0 : i32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %2 = aievec.upd %arg1[%arg4, %c1, %c2, %c1] {index = 0 : i8, offset = 0 : i32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %3 = aievec.upd %arg1[%arg4, %c2, %c2, %c0] {index = 0 : i8, offset = 0 : i32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %c0_5 = arith.constant 0 : index
//CHECK-NEXT:     %c254 = arith.constant 254 : index
//CHECK-NEXT:     %c1_6 = arith.constant 1 : index
//CHECK-NEXT:     scf.for %arg5 = %c0_5 to %c254 step %c1_6 {
//CHECK-NEXT:       %c1_7 = arith.constant 1 : index
//CHECK-NEXT:       %4 = arith.addi %arg5, %c1_7 : index
//CHECK-NEXT:       %c2_8 = arith.constant 2 : index
//CHECK-NEXT:       %5 = arith.addi %arg5, %c2_8 : index
//CHECK-NEXT:       %c0_9 = arith.constant 0 : index
//CHECK-NEXT:       %c254_10 = arith.constant 254 : index
//CHECK-NEXT:       %c8 = arith.constant 8 : index
//CHECK-NEXT:       scf.for %arg6 = %c0_9 to %c254_10 step %c8 {
//CHECK-NEXT:         %6 = aievec.upd %arg0[%arg3, %c0, %arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %7 = aievec.upd %arg2[%arg3, %arg4, %arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x10x254x254xf32>, vector<8xf32>
//CHECK-NEXT:         %8 = aievec_aie1.mac %6, %0, %7 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %c1_11 = arith.constant 1 : index
//CHECK-NEXT:         %9 = arith.addi %arg6, %c1_11 : index
//CHECK-NEXT:         %10 = aievec.upd %arg0[%arg3, %c0, %arg5, %9], %6 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %11 = aievec_aie1.mac %10, %0, %8 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %12 = aievec_aie1.mac %10, %0, %11 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %13 = aievec.upd %arg0[%arg3, %c0, %4, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %14 = aievec_aie1.mac %13, %0, %12 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %15 = aievec.upd %arg0[%arg3, %c0, %4, %9], %13 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %16 = aievec_aie1.mac %15, %0, %14 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %17 = aievec_aie1.mac %15, %0, %16 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %18 = aievec.upd %arg0[%arg3, %c0, %5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %19 = aievec_aie1.mac %18, %0, %17 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %20 = aievec.upd %arg0[%arg3, %c0, %5, %9], %18 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %21 = aievec_aie1.mac %20, %0, %19 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %22 = aievec_aie1.mac %20, %1, %21 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %23 = aievec.upd %arg0[%arg3, %c1, %arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %24 = aievec_aie1.mac %23, %1, %22 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %25 = aievec.upd %arg0[%arg3, %c1, %arg5, %9], %23 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %26 = aievec_aie1.mac %25, %1, %24 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %27 = aievec_aie1.mac %25, %1, %26 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %28 = aievec.upd %arg0[%arg3, %c1, %4, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %29 = aievec_aie1.mac %28, %1, %27 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %30 = aievec.upd %arg0[%arg3, %c1, %4, %9], %28 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %31 = aievec_aie1.mac %30, %1, %29 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %32 = aievec_aie1.mac %30, %1, %31 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %33 = aievec.upd %arg0[%arg3, %c1, %5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %34 = aievec_aie1.mac %33, %1, %32 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %35 = aievec.upd %arg0[%arg3, %c1, %5, %9], %33 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %36 = aievec_aie1.mac %35, %2, %34 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %37 = aievec_aie1.mac %35, %2, %36 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %38 = aievec.upd %arg0[%arg3, %c2, %arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %39 = aievec_aie1.mac %38, %2, %37 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %40 = aievec.upd %arg0[%arg3, %c2, %arg5, %9], %38 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %41 = aievec_aie1.mac %40, %2, %39 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %42 = aievec_aie1.mac %40, %2, %41 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %43 = aievec.upd %arg0[%arg3, %c2, %4, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %44 = aievec_aie1.mac %43, %2, %42 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %45 = aievec.upd %arg0[%arg3, %c2, %4, %9], %43 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %46 = aievec_aie1.mac %45, %2, %44 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %47 = aievec_aie1.mac %45, %2, %46 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %48 = aievec.upd %arg0[%arg3, %c2, %5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %49 = aievec_aie1.mac %48, %3, %47 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %50 = aievec.upd %arg0[%arg3, %c2, %5, %9], %48 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %51 = aievec_aie1.mac %50, %3, %49 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %52 = aievec_aie1.mac %50, %3, %51 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         vector.transfer_write %52, %arg2[%arg3, %arg4, %arg5, %arg6] {in_bounds = [true]} : vector<8xf32>, memref<10x10x254x254xf32>
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:   }
//CHECK-NEXT: }
