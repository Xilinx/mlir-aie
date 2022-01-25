// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=0" -split-input-file | FileCheck %s

//CHECK-LABEL: func @conv2d_0(%arg0: memref<?x?xi32>, %arg1: memref<?xi32>, %arg2: memref<?x?xi32>) {
func @conv2d_0 (%A: memref<?x?xi32>, %B: memref<?xi32>, %C: memref<?x?xi32>) {
    %c0 = arith.constant 0 : index
    %M = memref.dim %A, %c0 : memref<?x?xi32>
    %c1 = arith.constant 1 : index
    %N = memref.dim %A, %c1 : memref<?x?xi32>

    affine.for %arg3 = 0 to %M {
        affine.for %arg4 = 0 to %N {
            //Load the output point
            %ci = affine.load %C[%arg3, %arg4] : memref<?x?xi32>

            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<?x?xi32>
            %b11 = affine.load %B[0] : memref<?xi32>
            %p11 = arith.muli %a11, %b11 : i32
            %c11 = arith.addi %ci, %p11 : i32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<?x?xi32>
            %b12 = affine.load %B[1] : memref<?xi32>
            %p12 = arith.muli %a12, %b12 : i32
            %c12 = arith.addi %c11, %p12 : i32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<?x?xi32>
            %b13 = affine.load %B[2] : memref<?xi32>
            %p13 = arith.muli %a13, %b13 : i32
            %c13 = arith.addi %c12, %p13 : i32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<?x?xi32>
            %b21 = affine.load %B[3] : memref<?xi32>
            %p21 = arith.muli %a21, %b21 : i32
            %c21 = arith.addi %c13, %p21 : i32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<?x?xi32>
            %b22 = affine.load %B[4] : memref<?xi32>
            %p22 = arith.muli %a22, %b22 : i32
            %c22 = arith.addi %c21, %p22 : i32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<?x?xi32>
            %b23 = affine.load %B[5] : memref<?xi32>
            %p23 = arith.muli %a23, %b23 : i32
            %c23 = arith.addi %c22, %p23 : i32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<?x?xi32>
            %b31 = affine.load %B[6] : memref<?xi32>
            %p31 = arith.muli %a31, %b31 : i32
            %c31 = arith.addi %c23, %p31 : i32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<?x?xi32>
            %b32 = affine.load %B[7] : memref<?xi32>
            %p32 = arith.muli %a32, %b32 : i32
            %c32 = arith.addi %c31, %p32 : i32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<?x?xi32>
            %b33 = affine.load %B[8] : memref<?xi32>
            %p33 = arith.muli %a33, %b33 : i32
            %c33 = arith.addi %c32, %p33 : i32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<?x?xi32>
        }
    }
    return
}

//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: %c8 = arith.constant 8 : index
//CHECK-NEXT: %0 = memref.dim %arg0, %c0 : memref<?x?xi32>
//CHECK-NEXT: %1 = memref.dim %arg0, %c1 : memref<?x?xi32>
//CHECK-NEXT: %2 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<?xi32>, vector<8xi32>
//CHECK-NEXT: %3 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : si32} : memref<?xi32>, vector<8xi32>
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
//CHECK-NEXT: %6 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x?xi32>, vector<8xi32>
//CHECK-NEXT: %7 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x?xi32>, vector<16xi32>
//CHECK-NEXT: %8 = aievec.ups %6 {shift = 0 : i8} : vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %9 = aievec.mac %7, %2, %8 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %c1_5 = arith.constant 1 : index
//CHECK-NEXT: %10 = arith.addi %arg4, %c1_5 : index
//CHECK-NEXT: %11 = aievec.upd %arg0[%arg3, %10], %7 {index = 1 : i8, offset = 224 : si32} : memref<?x?xi32>, vector<16xi32>
//CHECK-NEXT: %12 = aievec.mac %11, %2, %9 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %13 = aievec.mac %11, %2, %12 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %14 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x?xi32>, vector<16xi32>
//CHECK-NEXT: %15 = aievec.mac %14, %2, %13 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %16 = aievec.upd %arg0[%4, %10], %14 {index = 1 : i8, offset = 224 : si32} : memref<?x?xi32>, vector<16xi32>
//CHECK-NEXT: %17 = aievec.mac %16, %2, %15 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %18 = aievec.mac %16, %2, %17 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %19 = aievec.upd %arg0[%5, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x?xi32>, vector<16xi32>
//CHECK-NEXT: %20 = aievec.mac %19, %2, %18 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %21 = aievec.upd %arg0[%5, %10], %19 {index = 1 : i8, offset = 224 : si32} : memref<?x?xi32>, vector<16xi32>
//CHECK-NEXT: %22 = aievec.mac %21, %2, %20 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %23 = aievec.mac %21, %3, %22 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %24 = aievec.srs %23 {shift = 0 : i8} : !aievec.acc<8xi80>, vector<8xi32>
//CHECK-NEXT: vector.transfer_write %24, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xi32>, memref<?x?xi32>


// CHECK-LABEL: func @conv2d_1(%arg0: memref<?x256xi32>, %arg1: memref<?xi32>, %arg2: memref<?x256xi32>) {
func @conv2d_1 (%A: memref<?x256xi32>, %B: memref<?xi32>, %C: memref<?x256xi32>) {
    %c0 = arith.constant 0 : index
    %M = memref.dim %A, %c0 : memref<?x256xi32>
    %c1 = arith.constant 1 : index
    %N = memref.dim %A, %c1 : memref<?x256xi32>

    affine.for %arg3 = 0 to %M {
        affine.for %arg4 = 0 to %N {
            //Load the output point
            %ci = affine.load %C[%arg3, %arg4] : memref<?x256xi32>

            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<?x256xi32>
            %b11 = affine.load %B[0] : memref<?xi32>
            %p11 = arith.muli %a11, %b11 : i32
            %c11 = arith.addi %ci, %p11 : i32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<?x256xi32>
            %b12 = affine.load %B[1] : memref<?xi32>
            %p12 = arith.muli %a12, %b12 : i32
            %c12 = arith.addi %c11, %p12 : i32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<?x256xi32>
            %b13 = affine.load %B[2] : memref<?xi32>
            %p13 = arith.muli %a13, %b13 : i32
            %c13 = arith.addi %c12, %p13 : i32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<?x256xi32>
            %b21 = affine.load %B[3] : memref<?xi32>
            %p21 = arith.muli %a21, %b21 : i32
            %c21 = arith.addi %c13, %p21 : i32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<?x256xi32>
            %b22 = affine.load %B[4] : memref<?xi32>
            %p22 = arith.muli %a22, %b22 : i32
            %c22 = arith.addi %c21, %p22 : i32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<?x256xi32>
            %b23 = affine.load %B[5] : memref<?xi32>
            %p23 = arith.muli %a23, %b23 : i32
            %c23 = arith.addi %c22, %p23 : i32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<?x256xi32>
            %b31 = affine.load %B[6] : memref<?xi32>
            %p31 = arith.muli %a31, %b31 : i32
            %c31 = arith.addi %c23, %p31 : i32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<?x256xi32>
            %b32 = affine.load %B[7] : memref<?xi32>
            %p32 = arith.muli %a32, %b32 : i32
            %c32 = arith.addi %c31, %p32 : i32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<?x256xi32>
            %b33 = affine.load %B[8] : memref<?xi32>
            %p33 = arith.muli %a33, %b33 : i32
            %c33 = arith.addi %c32, %p33 : i32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<?x256xi32>
        }
    }
    return
}

//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %c8 = arith.constant 8 : index
//CHECK-NEXT: %0 = memref.dim %arg0, %c0 : memref<?x256xi32>
//CHECK-NEXT: %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<?xi32>, vector<8xi32>
//CHECK-NEXT: %2 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : si32} : memref<?xi32>, vector<8xi32>
//CHECK-NEXT: %c0_0 = arith.constant 0 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: scf.for %arg3 = %c0_0 to %0 step %c1 {
//CHECK-NEXT: %c1_1 = arith.constant 1 : index
//CHECK-NEXT: %3 = arith.addi %arg3, %c1_1 : index
//CHECK-NEXT: %c2 = arith.constant 2 : index
//CHECK-NEXT: %4 = arith.addi %arg3, %c2 : index
//CHECK-NEXT: %c0_2 = arith.constant 0 : index
//CHECK-NEXT: %c256 = arith.constant 256 : index
//CHECK-NEXT: %c8_3 = arith.constant 8 : index
//CHECK-NEXT: scf.for %arg4 = %c0_2 to %c256 step %c8_3 {
//CHECK-NEXT: %5 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x256xi32>, vector<8xi32>
//CHECK-NEXT: %6 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x256xi32>, vector<16xi32>
//CHECK-NEXT: %7 = aievec.ups %5 {shift = 0 : i8} : vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %8 = aievec.mac %6, %1, %7 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %c1_4 = arith.constant 1 : index
//CHECK-NEXT: %9 = arith.addi %arg4, %c1_4 : index
//CHECK-NEXT: %10 = aievec.upd %arg0[%arg3, %9], %6 {index = 1 : i8, offset = 224 : si32} : memref<?x256xi32>, vector<16xi32>
//CHECK-NEXT: %11 = aievec.mac %10, %1, %8 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %12 = aievec.mac %10, %1, %11 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %13 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x256xi32>, vector<16xi32>
//CHECK-NEXT: %14 = aievec.mac %13, %1, %12 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %15 = aievec.upd %arg0[%3, %9], %13 {index = 1 : i8, offset = 224 : si32} : memref<?x256xi32>, vector<16xi32>
//CHECK-NEXT: %16 = aievec.mac %15, %1, %14 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %17 = aievec.mac %15, %1, %16 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %18 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<?x256xi32>, vector<16xi32>
//CHECK-NEXT: %19 = aievec.mac %18, %1, %17 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %20 = aievec.upd %arg0[%4, %9], %18 {index = 1 : i8, offset = 224 : si32} : memref<?x256xi32>, vector<16xi32>
//CHECK-NEXT: %21 = aievec.mac %20, %1, %19 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %22 = aievec.mac %20, %2, %21 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, !aievec.acc<8xi80>
//CHECK-NEXT: %23 = aievec.srs %22 {shift = 0 : i8} : !aievec.acc<8xi80>, vector<8xi32>
//CHECK-NEXT: vector.transfer_write %23, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xi32>, memref<?x256xi32>
