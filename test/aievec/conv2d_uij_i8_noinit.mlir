// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize="shift=10" -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
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

//CHECK-NEXT:    %c10_i32 = arith.constant 10 : i32
//CHECK-NEXT:    %c32 = arith.constant 32 : index
//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<48xi8>, vector<64xi8>
//CHECK-NEXT:    %1 = aievec.upd %arg1[%c32], %0 {index = 1 : i8, offset = 0 : i32} : memref<48xi8>, vector<64xi8>
//CHECK-NEXT:    %c0_0 = arith.constant 0 : index
//CHECK-NEXT:    %c16 = arith.constant 16 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    scf.for %arg3 = %c0_0 to %c16 step %c1 {
//CHECK-NEXT:      %c1_1 = arith.constant 1 : index
//CHECK-NEXT:      %2 = arith.addi %arg3, %c1_1 : index
//CHECK-NEXT:      %c2 = arith.constant 2 : index
//CHECK-NEXT:      %3 = arith.addi %arg3, %c2 : index
//CHECK-NEXT:      %c0_2 = arith.constant 0 : index
//CHECK-NEXT:      %c256 = arith.constant 256 : index
//CHECK-NEXT:      %c16_3 = arith.constant 16 : index
//CHECK-NEXT:      scf.for %arg4 = %c0_2 to %c256 step %c16_3 {
//CHECK-NEXT:        %4 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
//CHECK-NEXT:        %5 = aievec.mul %0, %4 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %6 = aievec.mul %0, %4 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "8", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %7 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
//CHECK-NEXT:        %8 = aievec.mac %0, %7, %5 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %9 = aievec.mac %0, %7, %6 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "8", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %10 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
//CHECK-NEXT:        %11 = aievec.mac %1, %10, %8 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %12 = aievec.mac %1, %10, %9 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "8", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %13 = aievec.srs %11, %c10_i32 : vector<16xi48>, i32, vector<16xi16>
//CHECK-NEXT:        %14 = aievec.srs %12, %c10_i32 : vector<16xi48>, i32, vector<16xi16>
//CHECK-NEXT:        %15 = aievec.concat %13, %14 : vector<16xi16>, vector<32xi16>
//CHECK-NEXT:        %16 = aievec.select %15 {select = "0xcccccccc", xoffsets = "0x0c080400", xoffsets_hi = "0x0", xsquare = "0x1010", xstart = "0", yoffsets = "0x0c080400", yoffsets_hi = "0x0", ysquare = "0x1010", ystart = "4"} : vector<32xi16>, vector<32xi16>
//CHECK-NEXT:        %17 = aievec.ext %16 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
//CHECK-NEXT:        %18 = aievec.pack %17 : vector<16xi16>, vector<16xi8>
//CHECK-NEXT:        vector.transfer_write %18, %arg2[%arg3, %arg4] : vector<16xi8>, memref<16x256xi8>
