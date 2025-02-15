// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" -aieml=true --aie-vectorize="shift=0 dup-factor=2" | aie-translate -aie2=true --aievec-to-cpp -o gen_aie-ml.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c %S/kernel.cc -o kernel.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/i8xi8.cc work/kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out"

// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" --convert-vector-to-aievec="aie-target=aie2" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o convert_aie-ml.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c %S/convert_kernel.cc -o convert_kernel.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/i8xi8.cc work/convert_kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out"

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
