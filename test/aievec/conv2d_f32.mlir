// RUN: aie-opt %s --affine-loop-unroll="unroll-full unroll-full-threshold=3" --canonicalize -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize -unaligned-loads-check=false -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<2048x2048xf32>, %arg1: memref<9xf32>, %arg2: memref<2046x2046xf32>) {
func.func @conv2d (%A: memref<2048x2048xf32>, %B: memref<9xf32>, %C: memref<2046x2046xf32>) {
    affine.for %arg3 = 0 to 2046 {
        affine.for %arg4 = 0 to 2046 {
            //3x3 stencil 
            affine.for %arg5 = 0 to 3 {
                affine.for %arg6 = 0 to 3 {   
                    //Load the output point
                    %ci = affine.load %C[%arg3, %arg4] : memref<2046x2046xf32>
                     %a11 = affine.load %A[%arg3+%arg5, %arg4+%arg6] : memref<2048x2048xf32>
                     %b11 = affine.load %B[3*%arg5+%arg6] : memref<9xf32>
                     %p11 = arith.mulf %a11, %b11 : f32
                     %c11 = arith.addf %ci, %p11 : f32
                     //Store accumulated sum
                     affine.store %c11, %C[%arg3, %arg4] : memref<2046x2046xf32>
                }
            }
        }
    }
    return
}

//CHECK-NEXT: %c8 = arith.constant 8 : index
//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT: %1 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : i32} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT: %c0_0 = arith.constant 0 : index
//CHECK-NEXT: %c2046 = arith.constant 2046 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: scf.for %arg3 = %c0_0 to %c2046 step %c1 {
//CHECK-NEXT: %c1_1 = arith.constant 1 : index
//CHECK-NEXT: %2 = arith.addi %arg3, %c1_1 : index
//CHECK-NEXT: %c2 = arith.constant 2 : index
//CHECK-NEXT: %3 = arith.addi %arg3, %c2 : index
//CHECK-NEXT: %c0_2 = arith.constant 0 : index
//CHECK-NEXT: %c2046_3 = arith.constant 2046 : index
//CHECK-NEXT: %c8_4 = arith.constant 8 : index
//CHECK-NEXT: scf.for %arg4 = %c0_2 to %c2046_3 step %c8_4 {
//CHECK-NEXT: %4 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2046x2046xf32>, vector<8xf32>
//CHECK-NEXT: %5 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %6 = aievec_aie1.mac %5, %0, %4 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %c1_5 = arith.constant 1 : index
//CHECK-NEXT: %7 = arith.addi %arg4, %c1_5 : index
//CHECK-NEXT: %8 = aievec.upd %arg0[%arg3, %7], %5 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %9 = aievec_aie1.mac %8, %0, %6 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %10 = aievec_aie1.mac %8, %0, %9 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %11 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %12 = aievec_aie1.mac %11, %0, %10 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %13 = aievec.upd %arg0[%2, %7], %11 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %14 = aievec_aie1.mac %13, %0, %12 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %15 = aievec_aie1.mac %13, %0, %14 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %16 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %17 = aievec_aie1.mac %16, %0, %15 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %18 = aievec.upd %arg0[%3, %7], %16 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %19 = aievec_aie1.mac %18, %0, %17 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %20 = aievec_aie1.mac %18, %1, %19 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: vector.transfer_write %20, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xf32>, memref<2046x2046xf32>
