// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize -unaligned-loads-check=false -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<2048x2048xi32>, %arg1: memref<3x3xi32>, %arg2: memref<2046x2046xi32>) {
func.func @conv2d (%A: memref<2048x2048xi32>, %B: memref<3x3xi32>, %C: memref<2046x2046xi32>) {
    affine.for %arg3 = 0 to 2046 {
        affine.for %arg4 = 0 to 2046 {
            affine.for %arg5 = 0 to 3 {
               //Load the output point
               %ci = affine.load %C[%arg3, %arg4] : memref<2046x2046xi32>

               //first point 
               %a1 = affine.load %A[%arg3+%arg5, %arg4+0] : memref<2048x2048xi32>
               %b1 = affine.load %B[%arg5, 0] : memref<3x3xi32>
               %p1 = arith.muli %a1, %b1 : i32
               %co1 = arith.addi %ci, %p1 : i32

               //second point 
               %a2 = affine.load %A[%arg3+%arg5, %arg4+1] : memref<2048x2048xi32>
               %b2 = affine.load %B[%arg5, 1] : memref<3x3xi32>
               %p2 = arith.muli %a2, %b2 : i32
               %co2 = arith.addi %co1, %p2 : i32

               //third point 
               %a3 = affine.load %A[%arg3+%arg5, %arg4+2] : memref<2048x2048xi32>
               %b3 = affine.load %B[%arg5, 2] : memref<3x3xi32>
               %p3 = arith.muli %a3, %b3 : i32
               %co3 = arith.addi %co2, %p3 : i32

               //Store accumulated sum
               affine.store %co3, %C[%arg3, %arg4] : memref<2046x2046xi32>
            }
        }
    }
    return
}

//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
//CHECK-NEXT:    %c0_0 = arith.constant 0 : index
//CHECK-NEXT:    %c2046 = arith.constant 2046 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    scf.for %arg3 = %c0_0 to %c2046 step %c1 {
//CHECK-NEXT:      %c0_1 = arith.constant 0 : index
//CHECK-NEXT:      %c2046_2 = arith.constant 2046 : index
//CHECK-NEXT:      %c8 = arith.constant 8 : index
//CHECK-NEXT:      scf.for %arg4 = %c0_1 to %c2046_2 step %c8 {
//CHECK-NEXT:        %0 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2046x2046xi32>, vector<8xi32>
//CHECK-NEXT:        %1 = aievec.ups %0 {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %c1_3 = arith.constant 1 : index
//CHECK-NEXT:        %2 = arith.addi %arg4, %c1_3 : index
//CHECK-NEXT:        %c0_4 = arith.constant 0 : index
//CHECK-NEXT:        %c3 = arith.constant 3 : index
//CHECK-NEXT:        %c1_5 = arith.constant 1 : index
//CHECK-NEXT:        scf.for %arg5 = %c0_4 to %c3 step %c1_5 {
//CHECK-NEXT:          %3 = arith.addi %arg3, %arg5 : index
//CHECK-NEXT:          %4 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi32>, vector<16xi32>
//CHECK-NEXT:          %5 = aievec.upd %arg1[%arg5, %c0] {index = 0 : i8, offset = 0 : i32} : memref<3x3xi32>, vector<8xi32>
//CHECK-NEXT:          %6 = aievec.mac %4, %5, %1 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:          %7 = aievec.upd %arg0[%3, %2], %4 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xi32>, vector<16xi32>
//CHECK-NEXT:          %8 = aievec.mac %7, %5, %6 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:          %9 = aievec.mac %7, %5, %8 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:          %10 = aievec.srs %9, %c0_i32 : vector<8xi80>, i32, vector<8xi32>
//CHECK-NEXT:          vector.transfer_write %10, %arg2[%arg3, %arg4] : vector<8xi32>, memref<2046x2046xi32>
