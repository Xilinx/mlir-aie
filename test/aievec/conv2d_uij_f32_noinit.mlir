// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=0" -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<2048x2048xf32>, %arg1: memref<9xf32>, %arg2: memref<2046x2046xf32>) {
func.func @conv2d (%A: memref<2048x2048xf32>, %B: memref<9xf32>, %C: memref<2046x2046xf32>) {
    affine.for %arg3 = 0 to 2046 {
        affine.for %arg4 = 0 to 2046 {
            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<2048x2048xf32>
            %b11 = affine.load %B[0] : memref<9xf32>
            %c11 = arith.mulf %a11, %b11 : f32

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
//CHECK-NEXT: %4 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %5 = aievec.mul %4, %0 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %c1_5 = arith.constant 1 : index
//CHECK-NEXT: %6 = arith.addi %arg4, %c1_5 : index
//CHECK-NEXT: %7 = aievec.upd %arg0[%arg3, %6], %4 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %8 = aievec.mac %7, %0, %5 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %9 = aievec.mac %7, %0, %8 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %10 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %11 = aievec.mac %10, %0, %9 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %12 = aievec.upd %arg0[%2, %6], %10 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %13 = aievec.mac %12, %0, %11 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %14 = aievec.mac %12, %0, %13 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %15 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %16 = aievec.mac %15, %0, %14 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %17 = aievec.upd %arg0[%3, %6], %15 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xf32>, vector<16xf32>
//CHECK-NEXT: %18 = aievec.mac %17, %0, %16 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %19 = aievec.mac %17, %1, %18 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: vector.transfer_write %19, %arg2[%arg3, %arg4] : vector<8xf32>, memref<2046x2046xf32>
