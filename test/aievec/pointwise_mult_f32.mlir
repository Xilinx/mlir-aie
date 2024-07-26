// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=10" -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @pointwise_mult(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>, %arg2: memref<2048xf32>) {
func.func @pointwise_mult (%A: memref<2048xf32>, %B: memref<2048xf32>, %C: memref<2048xf32>) {
    affine.for %arg0 = 0 to 2048 {
       %a = affine.load %A[%arg0] : memref<2048xf32>
       %b = affine.load %B[%arg0] : memref<2048xf32>
       //CHECK: %2 = aievec.concat %0, %0 : vector<8xf32>, vector<16xf32>
       //CHECK: %3 = aievec_aie1.mul %2, %1 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x76543210", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
       %c = arith.mulf %a, %b : f32
       affine.store %c, %C[%arg0] : memref<2048xf32>
    }
    return
}
