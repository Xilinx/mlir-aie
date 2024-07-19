// RUN: aie-opt %s --affine-loop-unroll="unroll-full unroll-full-threshold=3" --canonicalize -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize="zero-offset=4" -unaligned-loads-check=false -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<2048x2048xi16>, %arg1: memref<12xi16>, %arg2: memref<2046x2046xi16>) {
func.func @conv2d (%A: memref<2048x2048xi16>, %B: memref<12xi16>, %C: memref<2046x2046xi16>) {
    affine.for %arg3 = 0 to 2046 {
        affine.for %arg4 = 0 to 2046 {
            //3x3 stencil 
            affine.for %arg5 = 0 to 3 {
                affine.for %arg6 = 0 to 3 {   
                    //Load the output point
                    %ci = affine.load %C[%arg3, %arg4] : memref<2046x2046xi16>
                     %a11 = affine.load %A[%arg3+%arg5, %arg4+%arg6] : memref<2048x2048xi16>
                     %b11 = affine.load %B[4*%arg5+%arg6] : memref<12xi16>
                     %p11 = arith.muli %a11, %b11 : i16
                     %c11 = arith.addi %ci, %p11 : i16
                     //Store accumulated sum
                     affine.store %c11, %C[%arg3, %arg4] : memref<2046x2046xi16>
                }
            }
        }
    }
    return
}

//CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<12xi16>, vector<16xi16>
//CHECK-NEXT:    %c0_0 = arith.constant 0 : index
//CHECK-NEXT:    %c2046 = arith.constant 2046 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    scf.for %arg3 = %c0_0 to %c2046 step %c1 {
//CHECK-NEXT:      %c1_1 = arith.constant 1 : index
//CHECK-NEXT:      %1 = arith.addi %arg3, %c1_1 : index
//CHECK-NEXT:      %c2 = arith.constant 2 : index
//CHECK-NEXT:      %2 = arith.addi %arg3, %c2 : index
//CHECK-NEXT:      %c0_2 = arith.constant 0 : index
//CHECK-NEXT:      %c2046_3 = arith.constant 2046 : index
//CHECK-NEXT:      %c16 = arith.constant 16 : index
//CHECK-NEXT:      scf.for %arg4 = %c0_2 to %c2046_3 step %c16 {
//CHECK-NEXT:        %3 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2046x2046xi16>, vector<16xi16>
//CHECK-NEXT:        %4 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
//CHECK-NEXT:        %5 = aievec.upd %arg0[%arg3, %arg4], %4 {index = 1 : i8, offset = 256 : i32} : memref<2048x2048xi16>, vector<32xi16>
//CHECK-NEXT:        %6 = aievec.ups %3 {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
//CHECK-NEXT:        %7 = aievec_aie1.mac %5, %0, %6 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
//CHECK-NEXT:        %8 = aievec_aie1.mac %5, %0, %7 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
//CHECK-NEXT:        %9 = aievec.upd %arg0[%1, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
//CHECK-NEXT:        %10 = aievec.upd %arg0[%1, %arg4], %9 {index = 1 : i8, offset = 256 : i32} : memref<2048x2048xi16>, vector<32xi16>
//CHECK-NEXT:        %11 = aievec_aie1.mac %10, %0, %8 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
//CHECK-NEXT:        %12 = aievec_aie1.mac %10, %0, %11 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
//CHECK-NEXT:        %13 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048x2048xi16>, vector<32xi16>
//CHECK-NEXT:        %14 = aievec.upd %arg0[%2, %arg4], %13 {index = 1 : i8, offset = 256 : i32} : memref<2048x2048xi16>, vector<32xi16>
//CHECK-NEXT:        %15 = aievec_aie1.mac %14, %0, %12 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
//CHECK-NEXT:        %16 = aievec_aie1.mac %14, %0, %15 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x2110", xstart = "2", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"} : vector<32xi16>, vector<16xi16>, vector<16xi48>
//CHECK-NEXT:        %17 = aievec.srs %16, %c0_i32 : vector<16xi48>, i32, vector<16xi16>
//CHECK-NEXT:        vector.transfer_write %17, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<16xi16>, memref<2046x2046xi16>
