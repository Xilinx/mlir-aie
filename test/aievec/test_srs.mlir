// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<128xi32>, %arg1: memref<8xi32>, %arg2: memref<126xi32>) {
func.func @conv2d (%A: memref<128xi32>, %B: memref<8xi32>, %C: memref<126xi32>) {
    //CHECK-NEXT: %c0 = arith.constant 0 : index
    //CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
    //CHECK-NEXT: %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<8xi32>, vector<8xi32>
    //CHECK-NEXT: %c0_0 = arith.constant 0 : index
    //CHECK-NEXT: %c126 = arith.constant 126 : index
    //CHECK-NEXT: %c8 = arith.constant 8 : index
    //CHECK-NEXT: scf.for %arg3 = %c0_0 to %c126 step %c8 {
    affine.for %arg3 = 0 to 126 {
      //CHECK-NEXT: %1 = aievec.upd %arg2[%arg3] {index = 0 : i8, offset = 0 : i32} : memref<126xi32>, vector<8xi32>
      //CHECK-NEXT: %2 = aievec.upd %arg0[%arg3] {index = 0 : i8, offset = 0 : i32} : memref<128xi32>, vector<8xi32>
      //CHECK-NEXT: %3 = aievec.ups %1 {shift = 0 : i8} : vector<8xi32>, vector<8xi80>
      %ci = affine.load %C[%arg3] : memref<126xi32>
      %a = affine.load %A[%arg3] : memref<128xi32>
      %b = affine.load %B[0] : memref<8xi32>
      //CHECK-NEXT: %4 = aievec.concat %2, %2 : vector<8xi32>, vector<16xi32>
      //CHECK-NEXT: %5 = aievec_aie1.mac %4, %0, %3 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %p = arith.muli %a, %b : i32
      %co = arith.addi %ci, %p : i32
      //CHECK-NEXT: %6 = aievec.srs %5, %c0_i32 : vector<8xi80>, i32, vector<8xi32>
      //CHECK-NEXT: %7 = aievec_aie1.add %6, %6 : vector<8xi32>, vector<8xi32>, vector<8xi32>
      %co1 = arith.addi %co, %co : i32
      //CHECK-NEXT: vector.transfer_write %7, %arg2[%arg3] {in_bounds = [true]} : vector<8xi32>, memref<126xi32>
      affine.store %co1, %C[%arg3] : memref<126xi32>
    }
    return
}
