// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize="shift=0 zero-offset=4" -unaligned-loads-check=false -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<2048x2048xi16>, %arg1: memref<3x3xi16>, %arg2: memref<2046x2046xi16>) {
func.func @conv2d (%A: memref<2048x2048xi16>, %B: memref<3x3xi16>, %C: memref<2046x2046xi16>) {
    affine.for %arg3 = 0 to 2046 {
        affine.for %arg4 = 0 to 2046 {
            affine.for %arg5 = 0 to 3 {
               //Load the output point
               %ci = affine.load %C[%arg3, %arg4] : memref<2046x2046xi16>

               //first point 
               %a1 = affine.load %A[%arg3+%arg5, %arg4+0] : memref<2048x2048xi16>
               %b1 = affine.load %B[%arg5, 0] : memref<3x3xi16>
               %p1 = arith.muli %a1, %b1 : i16
               %co1 = arith.addi %ci, %p1 : i16

               //second point 
               %a2 = affine.load %A[%arg3+%arg5, %arg4+1] : memref<2048x2048xi16>
               %b2 = affine.load %B[%arg5, 1] : memref<3x3xi16>
               %p2 = arith.muli %a2, %b2 : i16
               %co2 = arith.addi %co1, %p2 : i16

               //third point 
               %a3 = affine.load %A[%arg3+%arg5, %arg4+2] : memref<2048x2048xi16>
               %b3 = affine.load %B[%arg5, 2] : memref<3x3xi16>
               %p3 = arith.muli %a3, %b3 : i16
               %co3 = arith.addi %co2, %p3 : i16

               //Store accumulated sum
               affine.store %co3, %C[%arg3, %arg4] : memref<2046x2046xi16>
            }
        }
    }
    return
}

//CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %c0_0 = arith.constant 0 : index
//CHECK-NEXT:    %c2046 = arith.constant 2046 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    scf.for %arg3 = %c0_0 to %c2046 step %c1 {
//CHECK-NEXT:      %c0_1 = arith.constant 0 : index
//CHECK-NEXT:      %c2046_2 = arith.constant 2046 : index
//CHECK-NEXT:      %c16 = arith.constant 16 : index
//CHECK-NEXT:      scf.for %arg4 = %c0_1 to %c2046_2 step %c16 {
//CHECK-NEXT:        %0 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2046x2046xi16>, vector<16xi16>
//CHECK-NEXT:        %1 = aievec.ups %0 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
//CHECK-NEXT:        %c0_3 = arith.constant 0 : index
//CHECK-NEXT:        %c3 = arith.constant 3 : index
//CHECK-NEXT:        %c1_4 = arith.constant 1 : index
//CHECK-NEXT:        scf.for %arg5 = %c0_3 to %c3 step %c1_4 {
//CHECK-NEXT:          %2 = arith.addi %arg3, %arg5 : index
//CHECK-NEXT:          %3 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
//CHECK-NEXT:          %4 = aievec.upd %arg0[%2, %arg4], %3 {index = 1 : i8, offset = 256 : i32} : memref<2048x2048xi16>, vector<32xi16>
//CHECK-NEXT:          %5 = aievec.upd %arg1[%arg5, %c0] {index = 0 : i8, offset = 0 : i32} : memref<3x3xi16>, vector<16xi16>
//CHECK-NEXT:          %6 = aievec_aie1.mac %4, %5, %1 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
//CHECK-NEXT:          %7 = aievec_aie1.mac %4, %5, %6 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
//CHECK-NEXT:          %8 = aievec.srs %7, %c0_i32 : vector<16xi48>, i32, vector<16xi16>
//CHECK-NEXT:          vector.transfer_write %8, %arg2[%arg3, %arg4] : vector<16xi16>, memref<2046x2046xi16>

