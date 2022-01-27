// RUN: aie-opt %s --convert-linalg-to-affine-loops --affine-loop-unroll="unroll-full unroll-full-threshold=3" --canonicalize -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize -split-input-file | FileCheck %s

//The affine dialect code is generated from the following linalg operator, with the maps inlined. 
//CHECK-LABEL: func @conv_2d(%arg0: memref<10x3x256x256xf32>, %arg1: memref<10x3x3x3xf32>, %arg2: memref<10x10x254x254xf32>) {
func @conv_2d(%input: memref<10x3x256x256xf32>, %filter: memref<10x3x3x3xf32>, %output: memref<10x10x254x254xf32>) {
  linalg.conv_2d_nchw_fchw{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<10x3x256x256xf32>, memref<10x3x3x3xf32>)
    outs (%output: memref<10x10x254x254xf32>)
  return
}

//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: %c2 = arith.constant 2 : index
//CHECK-NEXT: %c0_0 = arith.constant 0 : index
//CHECK-NEXT: %c10 = arith.constant 10 : index
//CHECK-NEXT: %c1_1 = arith.constant 1 : index
//CHECK-NEXT: scf.for %arg3 = %c0_0 to %c10 step %c1_1 {
//CHECK-NEXT:   %c0_2 = arith.constant 0 : index
//CHECK-NEXT:   %c10_3 = arith.constant 10 : index
//CHECK-NEXT:   %c1_4 = arith.constant 1 : index
//CHECK-NEXT:   scf.for %arg4 = %c0_2 to %c10_3 step %c1_4 {
//CHECK-NEXT:     %0 = aievec.upd %arg1[%arg4, %c0, %c0, %c0] {index = 0 : i8, offset = 0 : si32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %1 = aievec.upd %arg1[%arg4, %c0, %c2, %c2] {index = 0 : i8, offset = 0 : si32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %2 = aievec.upd %arg1[%arg4, %c1, %c2, %c1] {index = 0 : i8, offset = 0 : si32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %3 = aievec.upd %arg1[%arg4, %c2, %c2, %c0] {index = 0 : i8, offset = 0 : si32} : memref<10x3x3x3xf32>, vector<8xf32>
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
//CHECK-NEXT:         %6 = aievec.upd %arg0[%arg3, %c0, %arg5, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %7 = aievec.upd %arg2[%arg3, %arg4, %arg5, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x10x254x254xf32>, vector<8xf32>
//CHECK-NEXT:         %8 = aievec.ups %7 {shift = 0 : i8} : vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %9 = aievec.mac %6, %0, %8 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %c1_11 = arith.constant 1 : index
//CHECK-NEXT:         %10 = arith.addi %arg6, %c1_11 : index
//CHECK-NEXT:         %11 = aievec.upd %arg0[%arg3, %c0, %arg5, %10], %6 {index = 1 : i8, offset = 224 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %12 = aievec.mac %11, %0, %9 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %13 = aievec.mac %11, %0, %12 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %14 = aievec.upd %arg0[%arg3, %c0, %4, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %15 = aievec.mac %14, %0, %13 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %16 = aievec.upd %arg0[%arg3, %c0, %4, %10], %14 {index = 1 : i8, offset = 224 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %17 = aievec.mac %16, %0, %15 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %18 = aievec.mac %16, %0, %17 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %19 = aievec.upd %arg0[%arg3, %c0, %5, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %20 = aievec.mac %19, %0, %18 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %21 = aievec.upd %arg0[%arg3, %c0, %5, %10], %19 {index = 1 : i8, offset = 224 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %22 = aievec.mac %21, %0, %20 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %23 = aievec.mac %21, %1, %22 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %24 = aievec.upd %arg0[%arg3, %c1, %arg5, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %25 = aievec.mac %24, %1, %23 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %26 = aievec.upd %arg0[%arg3, %c1, %arg5, %10], %24 {index = 1 : i8, offset = 224 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %27 = aievec.mac %26, %1, %25 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %28 = aievec.mac %26, %1, %27 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %29 = aievec.upd %arg0[%arg3, %c1, %4, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %30 = aievec.mac %29, %1, %28 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %31 = aievec.upd %arg0[%arg3, %c1, %4, %10], %29 {index = 1 : i8, offset = 224 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %32 = aievec.mac %31, %1, %30 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %33 = aievec.mac %31, %1, %32 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %34 = aievec.upd %arg0[%arg3, %c1, %5, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %35 = aievec.mac %34, %1, %33 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %36 = aievec.upd %arg0[%arg3, %c1, %5, %10], %34 {index = 1 : i8, offset = 224 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %37 = aievec.mac %36, %2, %35 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %38 = aievec.mac %36, %2, %37 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %39 = aievec.upd %arg0[%arg3, %c2, %arg5, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %40 = aievec.mac %39, %2, %38 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %41 = aievec.upd %arg0[%arg3, %c2, %arg5, %10], %39 {index = 1 : i8, offset = 224 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %42 = aievec.mac %41, %2, %40 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %43 = aievec.mac %41, %2, %42 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %44 = aievec.upd %arg0[%arg3, %c2, %4, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %45 = aievec.mac %44, %2, %43 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %46 = aievec.upd %arg0[%arg3, %c2, %4, %10], %44 {index = 1 : i8, offset = 224 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %47 = aievec.mac %46, %2, %45 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %48 = aievec.mac %46, %2, %47 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %49 = aievec.upd %arg0[%arg3, %c2, %5, %arg6] {index = 0 : i8, offset = 0 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %50 = aievec.mac %49, %3, %48 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %51 = aievec.upd %arg0[%arg3, %c2, %5, %10], %49 {index = 1 : i8, offset = 224 : si32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %52 = aievec.mac %51, %3, %50 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %53 = aievec.mac %51, %3, %52 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
//CHECK-NEXT:         %54 = aievec.srs %53 {shift = 0 : i8} : !aievec.acc<8xf32>, vector<8xf32>
//CHECK-NEXT:         vector.transfer_write %54, %arg2[%arg3, %arg4, %arg5, %arg6] {in_bounds = [true]} : vector<8xf32>, memref<10x10x254x254xf32>
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:   }
//CHECK-NEXT: }
