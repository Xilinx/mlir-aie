// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=10" | aie-translate --aievec-to-cpp -o gen.cc
// RUN: xchesscc_wrapper aie -f -g +s +w work +o work -I%S -I. -c %S/kernel.cc -o kernel.o
// RUN: xchesscc_wrapper aie -f -g +s +w work +o work -I%S -I. %S/i32xi32.cc work/kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg -qf -T -P %aietools/data/versal_prod/lib -t "%S/../profiling.tcl ./work/a.out"
 
func.func @conv2d (%A: memref<18x272xi32>, %B: memref<9xi32>, %C: memref<16x256xi32>) {
    affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 256 {
            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<18x272xi32>
            %b11 = affine.load %B[0] : memref<9xi32>
            %c11 = arith.muli %a11, %b11 : i32

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<18x272xi32>
            %b12 = affine.load %B[1] : memref<9xi32>
            %p12 = arith.muli %a12, %b12 : i32
            %c12 = arith.addi %c11, %p12 : i32

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<18x272xi32>
            %b13 = affine.load %B[2] : memref<9xi32>
            %p13 = arith.muli %a13, %b13 : i32
            %c13 = arith.addi %c12, %p13 : i32

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<18x272xi32>
            %b21 = affine.load %B[3] : memref<9xi32>
            %p21 = arith.muli %a21, %b21 : i32
            %c21 = arith.addi %c13, %p21 : i32

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<18x272xi32>
            %b22 = affine.load %B[4] : memref<9xi32>
            %p22 = arith.muli %a22, %b22 : i32
            %c22 = arith.addi %c21, %p22 : i32

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<18x272xi32>
            %b23 = affine.load %B[5] : memref<9xi32>
            %p23 = arith.muli %a23, %b23 : i32
            %c23 = arith.addi %c22, %p23 : i32

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<18x272xi32>
            %b31 = affine.load %B[6] : memref<9xi32>
            %p31 = arith.muli %a31, %b31 : i32
            %c31 = arith.addi %c23, %p31 : i32

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<18x272xi32>
            %b32 = affine.load %B[7] : memref<9xi32>
            %p32 = arith.muli %a32, %b32 : i32
            %c32 = arith.addi %c31, %p32 : i32

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<18x272xi32>
            %b33 = affine.load %B[8] : memref<9xi32>
            %p33 = arith.muli %a33, %b33 : i32
            %c33 = arith.addi %c32, %p33 : i32

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<16x256xi32>
        }
    }
    return
}
