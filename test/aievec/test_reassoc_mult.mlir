// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @conv2d(%arg0: memref<256xi32>, %arg1: memref<1xi32>, %arg2: memref<256xi32>) {
func.func @conv2d (%A: memref<256xi32>, %B: memref<1xi32>, %C: memref<256xi32>) {
    affine.for %arg0 = 0 to 256 {
      %a1 = affine.load %A[%arg0] : memref<256xi32>
      %b1 = affine.load %B[0] : memref<1xi32>
      //CHECK: %2 = aievec.concat %1, %1 : vector<8xi32>, vector<16xi32>
      //CHECK: %3 = aievec_aie1.mul %2, %0 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xi32>, vector<8xi32>, vector<8xi80>
      %p1 = arith.muli %b1, %a1 : i32
      affine.store %p1, %C[%arg0] : memref<256xi32>
    }
    return
}
