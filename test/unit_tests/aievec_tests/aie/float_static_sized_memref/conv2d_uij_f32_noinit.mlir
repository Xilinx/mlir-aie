// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize | aie-translate --aievec-to-cpp -o gen.cc
// RUN: xchesscc_wrapper aie -f -g +s +w work +o work -I%S -I. -c %S/kernel.cc -o kernel.o
// RUN: xchesscc_wrapper aie -f -g +s +w work +o work -I%S -I. %S/float.cc work/kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg -qf -T -P %aietools/data/versal_prod/lib -t "%S/../../profiling.tcl ./work/a.out"

func.func @conv2d (%A: memref<18x272xf32>, %B: memref<9xf32>, %C: memref<16x256xf32>) {
    affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 256 {
            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<18x272xf32>
            %b11 = affine.load %B[0] : memref<9xf32>
            %c11 = arith.mulf %a11, %b11 : f32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<18x272xf32>
            %b12 = affine.load %B[1] : memref<9xf32>
            %p12 = arith.mulf %a12, %b12 : f32
            %c12 = arith.addf %c11, %p12 : f32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<18x272xf32>
            %b13 = affine.load %B[2] : memref<9xf32>
            %p13 = arith.mulf %a13, %b13 : f32
            %c13 = arith.addf %c12, %p13 : f32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<18x272xf32>
            %b21 = affine.load %B[3] : memref<9xf32>
            %p21 = arith.mulf %a21, %b21 : f32
            %c21 = arith.addf %c13, %p21 : f32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<18x272xf32>
            %b22 = affine.load %B[4] : memref<9xf32>
            %p22 = arith.mulf %a22, %b22 : f32
            %c22 = arith.addf %c21, %p22 : f32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<18x272xf32>
            %b23 = affine.load %B[5] : memref<9xf32>
            %p23 = arith.mulf %a23, %b23 : f32
            %c23 = arith.addf %c22, %p23 : f32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<18x272xf32>
            %b31 = affine.load %B[6] : memref<9xf32>
            %p31 = arith.mulf %a31, %b31 : f32
            %c31 = arith.addf %c23, %p31 : f32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<18x272xf32>
            %b32 = affine.load %B[7] : memref<9xf32>
            %p32 = arith.mulf %a32, %b32 : f32
            %c32 = arith.addf %c31, %p32 : f32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<18x272xf32>
            %b33 = affine.load %B[8] : memref<9xf32>
            %p33 = arith.mulf %a33, %b33 : f32
            %c33 = arith.addf %c32, %p33 : f32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<16x256xf32>
        }
    }
    return
}
