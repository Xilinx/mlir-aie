// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize="shift=10 zero-offset=4" | aie-translate --aievec-to-cpp -o gen.cc
// RUN: xchesscc_wrapper aie -f -g +s +w work +o work -I%S -I. -c %S/kernel.cc -o kernel.o
// RUN: xchesscc_wrapper aie -f -g +s +w work +o work -I%S -I. %S/i16xi16.cc work/kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg -qf -T -P %aietools/data/versal_prod/lib -t "%S/../../profiling.tcl ./work/a.out"

func.func @conv2d(%A: memref<?x?xi16>, %B: memref<?xi16>, %C: memref<?x?xi16>) {
    %c0 = arith.constant 0 : index
    %M = memref.dim %C, %c0 : memref<?x?xi16>
    %c1 = arith.constant 1 : index
    %N = memref.dim %C, %c1 : memref<?x?xi16>

    affine.for %arg3 = 0 to %M {
        affine.for %arg4 = 0 to %N {
            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<?x?xi16>
            %b11 = affine.load %B[0] : memref<?xi16>
            %p11 = arith.muli %a11, %b11 : i16

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<?x?xi16>
            %b12 = affine.load %B[1] : memref<?xi16>
            %p12 = arith.muli %a12, %b12 : i16
            %c12 = arith.addi %p11, %p12 : i16

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<?x?xi16>
            %b13 = affine.load %B[2] : memref<?xi16>
            %p13 = arith.muli %a13, %b13 : i16
            %c13 = arith.addi %c12, %p13 : i16

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<?x?xi16>
            %b21 = affine.load %B[4] : memref<?xi16>
            %p21 = arith.muli %a21, %b21 : i16
            %c21 = arith.addi %c13, %p21 : i16

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<?x?xi16>
            %b22 = affine.load %B[5] : memref<?xi16>
            %p22 = arith.muli %a22, %b22 : i16
            %c22 = arith.addi %c21, %p22 : i16

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<?x?xi16>
            %b23 = affine.load %B[6] : memref<?xi16>
            %p23 = arith.muli %a23, %b23 : i16
            %c23 = arith.addi %c22, %p23 : i16

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<?x?xi16>
            %b31 = affine.load %B[8] : memref<?xi16>
            %p31 = arith.muli %a31, %b31 : i16
            %c31 = arith.addi %c23, %p31 : i16

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<?x?xi16>
            %b32 = affine.load %B[9] : memref<?xi16>
            %p32 = arith.muli %a32, %b32 : i16
            %c32 = arith.addi %c31, %p32 : i16

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<?x?xi16>
            %b33 = affine.load %B[10] : memref<?xi16>
            %p33 = arith.muli %a33, %b33 : i16
            %c33 = arith.addi %c32, %p33 : i16

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<?x?xi16>
        }
    }
    return
}
