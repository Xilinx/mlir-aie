// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=0" -unaligned-loads-check=false -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<2048x2048xf32>, %arg1: memref<9xf32>, %arg2: memref<2046x2046xf32>) {
func.func @conv2d (%A: memref<2048x2048xf32>, %B: memref<9xf32>, %C: memref<2046x2046xf32>) {
    affine.for %arg3 = 0 to 2046 {
        affine.for %arg4 = 0 to 2046 {
            //Load the output point
            %ci = affine.load %C[%arg3, %arg4] : memref<2046x2046xf32>

            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<2048x2048xf32>
            %b11 = affine.load %B[0] : memref<9xf32>
            %p11 = arith.mulf %a11, %b11 : f32
            %c11 = arith.addf %ci, %p11 : f32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<2048x2048xf32>
            %b12 = affine.load %B[1] : memref<9xf32>
            %p12 = arith.mulf %a12, %b12 : f32
            %c12 = arith.addf %c11, %p12 : f32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<2048x2048xf32>
            %b13 = affine.load %B[2] : memref<9xf32>
            %p13 = arith.mulf %a13, %b13 : f32
            %c13 = arith.addf %c12, %p13 : f32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<2048x2048xf32>
            %b21 = affine.load %B[3] : memref<9xf32>
            %p21 = arith.mulf %a21, %b21 : f32
            %c21 = arith.addf %c13, %p21 : f32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<2048x2048xf32>
            %b22 = affine.load %B[4] : memref<9xf32>
            %p22 = arith.mulf %a22, %b22 : f32
            %c22 = arith.addf %c21, %p22 : f32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<2048x2048xf32>
            %b23 = affine.load %B[5] : memref<9xf32>
            %p23 = arith.mulf %a23, %b23 : f32
            %c23 = arith.addf %c22, %p23 : f32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<2048x2048xf32>
            %b31 = affine.load %B[6] : memref<9xf32>
            %p31 = arith.mulf %a31, %b31 : f32
            %c31 = arith.addf %c23, %p31 : f32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<2048x2048xf32>
            %b32 = affine.load %B[7] : memref<9xf32>
            %p32 = arith.mulf %a32, %b32 : f32
            %c32 = arith.addf %c31, %p32 : f32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<2048x2048xf32>
            %b33 = affine.load %B[8] : memref<9xf32>
            %p33 = arith.mulf %a33, %b33 : f32
            %c33 = arith.addf %c32, %p33 : f32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<2046x2046xf32>
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
//CHECK-NEXT: vector.transfer_write %20, %arg2[%arg3, %arg4] : vector<8xf32>, memref<2046x2046xf32>

// This test case will directly return the result generated from -affine-super-vectorize when
// -unaligned-loads-check=true. The reason is that in transfer_read %arg2[%arg3, %arg4],
// dim 1's memref shape size(2046) is not divisible by the vector lanes(8).

// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=0" -split-input-file 2>&1 | FileCheck %s -check-prefix=ALIGNMENT

// ALIGNMENT: vector.transfer_read's shape size of index 1 is not divisible by number of vector lanes.
// ALIGNMENT: Cannot apply aie-vectorize to func.func because alignment check has failed.                         
