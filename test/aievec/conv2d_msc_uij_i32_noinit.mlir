// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=0" -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<2048x2048xi32>, %arg1: memref<9xi32>, %arg2: memref<2046x2046xi32>) {
func.func @conv2d (%A: memref<2048x2048xi32>, %B: memref<9xi32>, %C: memref<2046x2046xi32>) {
    affine.for %arg3 = 0 to 2046 {
        affine.for %arg4 = 0 to 2046 {
            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<2048x2048xi32>
            %b11 = affine.load %B[0] : memref<9xi32>
            %c11 = arith.muli %a11, %b11 : i32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<2048x2048xi32>
            %b12 = affine.load %B[1] : memref<9xi32>
            %p12 = arith.muli %a12, %b12 : i32
            %c12 = arith.subi %c11, %p12 : i32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<2048x2048xi32>
            %b13 = affine.load %B[2] : memref<9xi32>
            %p13 = arith.muli %a13, %b13 : i32
            %c13 = arith.subi %c12, %p13 : i32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<2048x2048xi32>
            %b21 = affine.load %B[3] : memref<9xi32>
            %p21 = arith.muli %a21, %b21 : i32
            %c21 = arith.subi %c13, %p21 : i32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<2048x2048xi32>
            %b22 = affine.load %B[4] : memref<9xi32>
            %p22 = arith.muli %a22, %b22 : i32
            %c22 = arith.subi %c21, %p22 : i32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<2048x2048xi32>
            %b23 = affine.load %B[5] : memref<9xi32>
            %p23 = arith.muli %a23, %b23 : i32
            %c23 = arith.subi %c22, %p23 : i32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<2048x2048xi32>
            %b31 = affine.load %B[6] : memref<9xi32>
            %p31 = arith.muli %a31, %b31 : i32
            %c31 = arith.subi %c23, %p31 : i32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<2048x2048xi32>
            %b32 = affine.load %B[7] : memref<9xi32>
            %p32 = arith.muli %a32, %b32 : i32
            %c32 = arith.subi %c31, %p32 : i32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<2048x2048xi32>
            %b33 = affine.load %B[8] : memref<9xi32>
            %p33 = arith.muli %a33, %b33 : i32
            %c33 = arith.subi %c32, %p33 : i32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<2046x2046xi32>
        }
    }
    return
}

//CHECK-NEXT:    %c8 = arith.constant 8 : index
//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
//CHECK-NEXT:    %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<9xi32>, vector<8xi32>
//CHECK-NEXT:    %1 = aievec.upd %arg1[%c8] {index = 0 : i8, offset = 0 : i32} : memref<9xi32>, vector<8xi32>
//CHECK-NEXT:    %c0_0 = arith.constant 0 : index
//CHECK-NEXT:    %c2046 = arith.constant 2046 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    scf.for %arg3 = %c0_0 to %c2046 step %c1 {
//CHECK-NEXT:      %c1_1 = arith.constant 1 : index
//CHECK-NEXT:      %2 = arith.addi %arg3, %c1_1 : index
//CHECK-NEXT:      %c2 = arith.constant 2 : index
//CHECK-NEXT:      %3 = arith.addi %arg3, %c2 : index
//CHECK-NEXT:      %c0_2 = arith.constant 0 : index
//CHECK-NEXT:      %c2046_3 = arith.constant 2046 : index
//CHECK-NEXT:      %c8_4 = arith.constant 8 : index
//CHECK-NEXT:      scf.for %arg4 = %c0_2 to %c2046_3 step %c8_4 {
//CHECK-NEXT:        %4 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi32>, vector<16xi32>
//CHECK-NEXT:        %5 = aievec_aie1.mul %4, %0 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %c1_5 = arith.constant 1 : index
//CHECK-NEXT:        %6 = arith.addi %arg4, %c1_5 : index
//CHECK-NEXT:        %7 = aievec.upd %arg0[%arg3, %6], %4 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xi32>, vector<16xi32>
//CHECK-NEXT:        %8 = aievec_aie1.mac %7, %0, %5 {fmsub = true, xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %9 = aievec_aie1.mac %7, %0, %8 {fmsub = true, xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %10 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi32>, vector<16xi32>
//CHECK-NEXT:        %11 = aievec_aie1.mac %10, %0, %9 {fmsub = true, xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %12 = aievec.upd %arg0[%2, %6], %10 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xi32>, vector<16xi32>
//CHECK-NEXT:        %13 = aievec_aie1.mac %12, %0, %11 {fmsub = true, xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %14 = aievec_aie1.mac %12, %0, %13 {fmsub = true, xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %15 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi32>, vector<16xi32>
//CHECK-NEXT:        %16 = aievec_aie1.mac %15, %0, %14 {fmsub = true, xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %17 = aievec.upd %arg0[%3, %6], %15 {index = 1 : i8, offset = 224 : i32} : memref<2048x2048xi32>, vector<16xi32>
//CHECK-NEXT:        %18 = aievec_aie1.mac %17, %0, %16 {fmsub = true, xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %19 = aievec_aie1.mac %17, %1, %18 {fmsub = true, xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
//CHECK-NEXT:        %20 = aievec.srs %19, %c0_i32 : vector<8xi80>, i32, vector<8xi32>
//CHECK-NEXT:        vector.transfer_write %20, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xi32>, memref<2046x2046xi32>
