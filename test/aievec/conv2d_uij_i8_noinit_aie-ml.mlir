// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" -aieml=true --aie-vectorize="shift=0 dup-factor=2" | FileCheck %s

// CHECK-LABEL: func.func @conv2d(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
func.func @conv2d (%A: memref<18x288xi8>, %B: memref<48xi8>, %C: memref<16x256xi8>) {
    affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 256 {
            //First row
            //first point
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<18x288xi8>
            %b11 = affine.load %B[0] : memref<48xi8>
            %p11 = arith.muli %a11, %b11 : i8

            //second point
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<18x288xi8>
            %b12 = affine.load %B[2] : memref<48xi8>
            %p12 = arith.muli %a12, %b12 : i8
            %c12 = arith.addi %p11, %p12 : i8

            //third point
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<18x288xi8>
            %b13 = affine.load %B[4] : memref<48xi8>
            %p13 = arith.muli %a13, %b13 : i8
            %c13 = arith.addi %c12, %p13 : i8

            //Second row
            //first point
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<18x288xi8>
            %b21 = affine.load %B[16] : memref<48xi8>
            %p21 = arith.muli %a21, %b21 : i8
            %c21 = arith.addi %c13, %p21 : i8

            //second point
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<18x288xi8>
            %b22 = affine.load %B[18] : memref<48xi8>
            %p22 = arith.muli %a22, %b22 : i8
            %c22 = arith.addi %c21, %p22 : i8

            //third point
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<18x288xi8>
            %b23 = affine.load %B[20] : memref<48xi8>
            %p23 = arith.muli %a23, %b23 : i8
            %c23 = arith.addi %c22, %p23 : i8

            //Third row
            //first point
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<18x288xi8>
            %b31 = affine.load %B[32] : memref<48xi8>
            %p31 = arith.muli %a31, %b31 : i8
            %c31 = arith.addi %c23, %p31 : i8

            //second point
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<18x288xi8>
            %b32 = affine.load %B[34] : memref<48xi8>
            %p32 = arith.muli %a32, %b32 : i8
            %c32 = arith.addi %c31, %p32 : i8

            //third point
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<18x288xi8>
            %b33 = affine.load %B[36] : memref<48xi8>
            %p33 = arith.muli %a33, %b33 : i8
            %c33 = arith.addi %c32, %p33 : i8

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<16x256xi8>
        }
    }
    return
}

//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<48xi8>, vector<64xi8>
//CHECK-NEXT:    %1 = aievec.shuffle %0 {mode = 0 : i32} : vector<64xi8>, vector<64xi8>
//CHECK-NEXT:    %2 = aievec.shift %1 {shift = 8 : i32} : vector<64xi8>, vector<64xi8>
//CHECK-NEXT:    %3 = aievec.shift %1 {shift = 16 : i32} : vector<64xi8>, vector<64xi8>
//CHECK-NEXT:    %c0_0 = arith.constant 0 : index
//CHECK-NEXT:    %c16 = arith.constant 16 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    scf.for %arg3 = %c0_0 to %c16 step %c1 {
//CHECK-NEXT:      %c1_1 = arith.constant 1 : index
//CHECK-NEXT:      %4 = arith.addi %arg3, %c1_1 : index
//CHECK-NEXT:      %c2 = arith.constant 2 : index
//CHECK-NEXT:      %5 = arith.addi %arg3, %c2 : index
//CHECK-NEXT:      %c0_2 = arith.constant 0 : index
//CHECK-NEXT:      %c256 = arith.constant 256 : index
//CHECK-NEXT:      %c32 = arith.constant 32 : index
//CHECK-NEXT:      scf.for %arg4 = %c0_2 to %c256 step %c32 {
//CHECK-NEXT:        %6 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<64xi8>
//CHECK-NEXT:        %7 = aievec.mul_conv %6, %1 {M = 32 : i32, N = 8 : i32} : vector<64xi8>, vector<64xi8>, vector<32xi32>
//CHECK-NEXT:        %8 = aievec.upd %arg0[%4, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<64xi8>
//CHECK-NEXT:        %9 = aievec.fma_conv %8, %2, %7 {M = 32 : i32, N = 8 : i32} : vector<64xi8>, vector<64xi8>, vector<32xi32>
//CHECK-NEXT:        %10 = aievec.upd %arg0[%5, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<64xi8>
//CHECK-NEXT:        %11 = aievec.fma_conv %10, %3, %9 {M = 32 : i32, N = 8 : i32} : vector<64xi8>, vector<64xi8>, vector<32xi32>
//CHECK-NEXT:        %12 = aievec.srs %11 {shift = 0 : i8} : vector<32xi32>, vector<32xi8>
//CHECK-NEXT:        vector.transfer_write %12, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<32xi8>, memref<16x256xi8>
