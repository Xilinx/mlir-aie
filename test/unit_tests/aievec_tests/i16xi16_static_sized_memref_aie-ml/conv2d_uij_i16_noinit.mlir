// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" -aieml=true --aie-vectorize="shift=10 zero-offset=4" | aie-translate -aie2=true --aievec-to-cpp -o gen_aie-ml.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c %S/kernel.cc -o kernel.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc work/kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out"

// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --convert-vector-to-aievec="aie-target=aie2 shift=10" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o convert_aie-ml.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c %S/convert_kernel.cc -o convert_kernel.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc work/convert_kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out"

func.func @conv2d (%A: memref<18x288xi16>, %B: memref<9xi16>, %C: memref<16x256xi16>) {
    affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 256 {
            //First row
            //first point 
            %a11 = affine.load %A[%arg3, %arg4+0] : memref<18x288xi16>
            %b11 = affine.load %B[0] : memref<9xi16>
            %p11 = arith.muli %a11, %b11 : i16

            //second point 
            %a12 = affine.load %A[%arg3, %arg4+1] : memref<18x288xi16>
            %b12 = affine.load %B[1] : memref<9xi16>
            %p12 = arith.muli %a12, %b12 : i16
            %c12 = arith.addi %p11, %p12 : i16

            //third point 
            %a13 = affine.load %A[%arg3, %arg4+2] : memref<18x288xi16>
            %b13 = affine.load %B[2] : memref<9xi16>
            %p13 = arith.muli %a13, %b13 : i16
            %c13 = arith.addi %c12, %p13 : i16

            //Second row
            //first point 
            %a21 = affine.load %A[%arg3+1, %arg4+0] : memref<18x288xi16>
            %b21 = affine.load %B[4] : memref<9xi16>
            %p21 = arith.muli %a21, %b21 : i16
            %c21 = arith.addi %c13, %p21 : i16

            //second point 
            %a22 = affine.load %A[%arg3+1, %arg4+1] : memref<18x288xi16>
            %b22 = affine.load %B[5] : memref<9xi16>
            %p22 = arith.muli %a22, %b22 : i16
            %c22 = arith.addi %c21, %p22 : i16

            //third point 
            %a23 = affine.load %A[%arg3+1, %arg4+2] : memref<18x288xi16>
            %b23 = affine.load %B[6] : memref<9xi16>
            %p23 = arith.muli %a23, %b23 : i16
            %c23 = arith.addi %c22, %p23 : i16

            //Third row
            //first point 
            %a31 = affine.load %A[%arg3+2, %arg4+0] : memref<18x288xi16>
            %b31 = affine.load %B[8] : memref<9xi16>
            %p31 = arith.muli %a31, %b31 : i16
            %c31 = arith.addi %c23, %p31 : i16

            //second point 
            %a32 = affine.load %A[%arg3+2, %arg4+1] : memref<18x288xi16>
            %b32 = affine.load %B[9] : memref<9xi16>
            %p32 = arith.muli %a32, %b32 : i16
            %c32 = arith.addi %c31, %p32 : i16

            //third point 
            %a33 = affine.load %A[%arg3+2, %arg4+2] : memref<18x288xi16>
            %b33 = affine.load %B[10] : memref<9xi16>
            %p33 = arith.muli %a33, %b33 : i16
            %c33 = arith.addi %c32, %p33 : i16

            //Store accumulated sum
            affine.store %c33, %C[%arg3, %arg4] : memref<16x256xi16>
        }
    }
    return
}
