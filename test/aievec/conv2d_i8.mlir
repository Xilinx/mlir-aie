// RUN: aie-opt %s --affine-loop-unroll="unroll-full unroll-full-threshold=3" --canonicalize -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @conv2d(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
func.func @conv2d (%A: memref<18x288xi8>, %B: memref<48xi8>, %C: memref<16x256xi8>) {
    affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 256 {
            //3x3 stencil 
            affine.for %arg5 = 0 to 3 {
                affine.for %arg6 = 0 to 3 {   
                    //Load the output point
                    %ci = affine.load %C[%arg3, %arg4] : memref<16x256xi8>
                     %a11 = affine.load %A[%arg3+%arg5, %arg4+%arg6] : memref<18x288xi8>
                     %b11 = affine.load %B[16*%arg5 + 2*%arg6] : memref<48xi8>
                     %p11 = arith.muli %a11, %b11 : i8
                     %c11 = arith.addi %ci, %p11 : i8
                     //Store accumulated sum
                     affine.store %c11, %C[%arg3, %arg4] : memref<16x256xi8>
                }
            }
        }
    }
    return
}

//CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
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
//CHECK-NEXT:        %4 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<16x256xi8>, vector<16xi8>
//CHECK-NEXT:        %5 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
//CHECK-NEXT:        %6 = aievec.ups %4 {shift = 0 : i8} : vector<16xi8>, vector<16xi48>
//CHECK-NEXT:        %7 = aievec_aie1.mac %0, %5, %6 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %8 = aievec_aie1.mac %0, %5, %6 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "8", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %9 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
//CHECK-NEXT:        %10 = aievec_aie1.mac %0, %9, %7 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %11 = aievec_aie1.mac %0, %9, %8 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "8", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %12 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<32xi8>
//CHECK-NEXT:        %13 = aievec_aie1.mac %1, %12, %10 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %14 = aievec_aie1.mac %1, %12, %11 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "8", zstep = "2"} : vector<64xi8>, vector<32xi8>, vector<16xi48>
//CHECK-NEXT:        %15 = aievec.srs %13, %c0_i32 : vector<16xi48>, i32, vector<16xi16>
//CHECK-NEXT:        %16 = aievec.srs %14, %c0_i32 : vector<16xi48>, i32, vector<16xi16>
//CHECK-NEXT:        %17 = aievec.concat %15, %16 : vector<16xi16>, vector<32xi16>
//CHECK-NEXT:        %18 = aievec.select %17 {select = "0xcccccccc", xoffsets = "0x0c080400", xoffsets_hi = "0x0", xsquare = "0x1010", xstart = "0", yoffsets = "0x0c080400", yoffsets_hi = "0x0", ysquare = "0x1010", ystart = "4"} : vector<32xi16>, vector<32xi16>
//CHECK-NEXT:        %19 = aievec.ext %18 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
//CHECK-NEXT:        %20 = aievec.pack %19 : vector<16xi16>, vector<16xi8>
//CHECK-NEXT:        vector.transfer_write %20, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<16xi8>, memref<16x256xi8>

