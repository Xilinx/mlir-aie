// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=0" -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @conv2d_0(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>, %arg2: memref<?x?xf32>) {
func.func @conv2d_0 (%A: memref<?x?xf32>, %B: memref<?xf32>, %C: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %M = memref.dim %A, %c0 : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %N = memref.dim %A, %c1 : memref<?x?xf32>

    affine.for %arg3 = 0 to %M {
        affine.for %arg4 = 0 to %N {
            //Load the output point
            %ci = affine.load %C[%arg3, %arg4] : memref<?x?xf32>

            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<?x?xf32>
            %b11 = affine.load %B[0] : memref<?xf32>
            %p11 = arith.mulf %a11, %b11 : f32
            %c11 = arith.addf %ci, %p11 : f32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<?x?xf32>
            %b12 = affine.load %B[1] : memref<?xf32>
            %p12 = arith.mulf %a12, %b12 : f32
            %c12 = arith.addf %c11, %p12 : f32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<?x?xf32>
            %b13 = affine.load %B[2] : memref<?xf32>
            %p13 = arith.mulf %a13, %b13 : f32
            %c13 = arith.addf %c12, %p13 : f32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<?x?xf32>
            %b21 = affine.load %B[3] : memref<?xf32>
            %p21 = arith.mulf %a21, %b21 : f32
            %c21 = arith.addf %c13, %p21 : f32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<?x?xf32>
            %b22 = affine.load %B[4] : memref<?xf32>
            %p22 = arith.mulf %a22, %b22 : f32
            %c22 = arith.addf %c21, %p22 : f32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<?x?xf32>
            %b23 = affine.load %B[5] : memref<?xf32>
            %p23 = arith.mulf %a23, %b23 : f32
            %c23 = arith.addf %c22, %p23 : f32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<?x?xf32>
            %b31 = affine.load %B[6] : memref<?xf32>
            %p31 = arith.mulf %a31, %b31 : f32
            %c31 = arith.addf %c23, %p31 : f32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<?x?xf32>
            %b32 = affine.load %B[7] : memref<?xf32>
            %p32 = arith.mulf %a32, %b32 : f32
            %c32 = arith.addf %c31, %p32 : f32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<?x?xf32>
            %b33 = affine.load %B[8] : memref<?xf32>
            %p33 = arith.mulf %a33, %b33 : f32
            %c33 = arith.addf %c32, %p33 : f32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<?x?xf32>
        }
    }
    return
}

//CHECK-NEXT: %c8 = arith.constant 8 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
//CHECK-NEXT: %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
//CHECK-NEXT: %2 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<?xf32>, vector<8xf32>
//CHECK-NEXT: %3 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : si32} : memref<?xf32>, vector<8xf32>
//CHECK-NEXT: %c0_0 = arith.constant 0 : index
//CHECK-NEXT: %c1_1 = arith.constant 1 : index
//CHECK-NEXT: scf.for %arg3 = %c0_0 to %0 step %c1_1 {
//CHECK-NEXT: %c1_2 = arith.constant 1 : index
//CHECK-NEXT: %4 = arith.addi %arg3, %c1_2 : index
//CHECK-NEXT: %c2 = arith.constant 2 : index
//CHECK-NEXT: %5 = arith.addi %arg3, %c2 : index
//CHECK-NEXT: %c0_3 = arith.constant 0 : index
//CHECK-NEXT: %c8_4 = arith.constant 8 : index
//CHECK-NEXT: scf.for %arg4 = %c0_3 to %1 step %c8_4 {
//CHECK-NEXT: %6 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x?xf32>, vector<8xf32>
//CHECK-NEXT: %7 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x?xf32>, vector<16xf32>
//CHECK-NEXT: %8 = aievec.mac %7, %2, %6 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %c1_5 = arith.constant 1 : index
//CHECK-NEXT: %9 = arith.addi %arg4, %c1_5 : index
//CHECK-NEXT: %10 = aievec.upd %arg0[%arg3, %9], %7 {index = 1 : i8, offset = 224 : si32} : memref<?x?xf32>, vector<16xf32>
//CHECK-NEXT: %11 = aievec.mac %10, %2, %8 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %12 = aievec.mac %10, %2, %11 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %13 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x?xf32>, vector<16xf32>
//CHECK-NEXT: %14 = aievec.mac %13, %2, %12 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %15 = aievec.upd %arg0[%4, %9], %13 {index = 1 : i8, offset = 224 : si32} : memref<?x?xf32>, vector<16xf32>
//CHECK-NEXT: %16 = aievec.mac %15, %2, %14 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %17 = aievec.mac %15, %2, %16 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %18 = aievec.upd %arg0[%5, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x?xf32>, vector<16xf32>
//CHECK-NEXT: %19 = aievec.mac %18, %2, %17 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %20 = aievec.upd %arg0[%5, %9], %18 {index = 1 : i8, offset = 224 : si32} : memref<?x?xf32>, vector<16xf32>
//CHECK-NEXT: %21 = aievec.mac %20, %2, %19 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: %22 = aievec.mac %20, %3, %21 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT: vector.transfer_write %22, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xf32>, memref<?x?xf32>


//CHECK-LABEL: func.func @conv2d_1(%arg0: memref<?x256xf32>, %arg1: memref<?xf32>, %arg2: memref<?x256xf32>) {
func.func @conv2d_1 (%A: memref<?x256xf32>, %B: memref<?xf32>, %C: memref<?x256xf32>) {
    %c0 = arith.constant 0 : index
    %M = memref.dim %A, %c0 : memref<?x256xf32>
    %c1 = arith.constant 1 : index
    %N = memref.dim %A, %c1 : memref<?x256xf32>

    affine.for %arg3 = 0 to %M {
        affine.for %arg4 = 0 to %N {
            //Load the output point
            %ci = affine.load %C[%arg3, %arg4] : memref<?x256xf32>

            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<?x256xf32>
            %b11 = affine.load %B[0] : memref<?xf32>
            %p11 = arith.mulf %a11, %b11 : f32
            %c11 = arith.addf %ci, %p11 : f32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<?x256xf32>
            %b12 = affine.load %B[1] : memref<?xf32>
            %p12 = arith.mulf %a12, %b12 : f32
            %c12 = arith.addf %c11, %p12 : f32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<?x256xf32>
            %b13 = affine.load %B[2] : memref<?xf32>
            %p13 = arith.mulf %a13, %b13 : f32
            %c13 = arith.addf %c12, %p13 : f32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<?x256xf32>
            %b21 = affine.load %B[3] : memref<?xf32>
            %p21 = arith.mulf %a21, %b21 : f32
            %c21 = arith.addf %c13, %p21 : f32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<?x256xf32>
            %b22 = affine.load %B[4] : memref<?xf32>
            %p22 = arith.mulf %a22, %b22 : f32
            %c22 = arith.addf %c21, %p22 : f32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<?x256xf32>
            %b23 = affine.load %B[5] : memref<?xf32>
            %p23 = arith.mulf %a23, %b23 : f32
            %c23 = arith.addf %c22, %p23 : f32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<?x256xf32>
            %b31 = affine.load %B[6] : memref<?xf32>
            %p31 = arith.mulf %a31, %b31 : f32
            %c31 = arith.addf %c23, %p31 : f32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<?x256xf32>
            %b32 = affine.load %B[7] : memref<?xf32>
            %p32 = arith.mulf %a32, %b32 : f32
            %c32 = arith.addf %c31, %p32 : f32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<?x256xf32>
            %b33 = affine.load %B[8] : memref<?xf32>
            %p33 = arith.mulf %a33, %b33 : f32
            %c33 = arith.addf %c32, %p33 : f32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<?x256xf32>
        }
    }
    return
}

//CHECK-NEXT %c8 = arith.constant 8 : index
//CHECK-NEXT %c0 = arith.constant 0 : index
//CHECK-NEXT %0 = memref.dim %arg0, %c0 : memref<?x256xf32>
//CHECK-NEXT %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<?xf32>, vector<8xf32>
//CHECK-NEXT %2 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : si32} : memref<?xf32>, vector<8xf32>
//CHECK-NEXT %c0_0 = arith.constant 0 : index
//CHECK-NEXT %c1 = arith.constant 1 : index
//CHECK-NEXT scf.for %arg3 = %c0_0 to %0 step %c1 {
//CHECK-NEXT %c1_1 = arith.constant 1 : index
//CHECK-NEXT %3 = arith.addi %arg3, %c1_1 : index
//CHECK-NEXT %c2 = arith.constant 2 : index
//CHECK-NEXT %4 = arith.addi %arg3, %c2 : index
//CHECK-NEXT %c0_2 = arith.constant 0 : index
//CHECK-NEXT %c256 = arith.constant 256 : index
//CHECK-NEXT %c8_3 = arith.constant 8 : index
//CHECK-NEXT scf.for %arg4 = %c0_2 to %c256 step %c8_3 {
//CHECK-NEXT %5 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x256xf32>, vector<8xf32>
//CHECK-NEXT %6 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x256xf32>, vector<16xf32>
//CHECK-NEXT %7 = aievec.mac %6, %1, %5 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT %c1_4 = arith.constant 1 : index
//CHECK-NEXT %8 = arith.addi %arg4, %c1_4 : index
//CHECK-NEXT %9 = aievec.upd %arg0[%arg3, %8], %6 {index = 1 : i8, offset = 224 : si32} : memref<?x256xf32>, vector<16xf32>
//CHECK-NEXT %10 = aievec.mac %9, %1, %7 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT %11 = aievec.mac %9, %1, %10 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT %12 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x256xf32>, vector<16xf32>
//CHECK-NEXT %13 = aievec.mac %12, %1, %11 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT %14 = aievec.upd %arg0[%3, %8], %12 {index = 1 : i8, offset = 224 : si32} : memref<?x256xf32>, vector<16xf32>
//CHECK-NEXT %15 = aievec.mac %14, %1, %13 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT %16 = aievec.mac %14, %1, %15 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT %17 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x256xf32>, vector<16xf32>
//CHECK-NEXT %18 = aievec.mac %17, %1, %16 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT %19 = aievec.upd %arg0[%4, %8], %17 {index = 1 : i8, offset = 224 : si32} : memref<?x256xf32>, vector<16xf32>
//CHECK-NEXT %20 = aievec.mac %19, %1, %18 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT %21 = aievec.mac %19, %2, %20 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT vector.transfer_write %21, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xf32>, memref<?x256xf32>
