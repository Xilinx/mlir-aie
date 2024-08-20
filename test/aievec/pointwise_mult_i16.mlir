// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --aie-vectorize="shift=10" -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @pointwise_mult(%arg0: memref<2048xi16>, %arg1: memref<2048xi16>, %arg2: memref<2048xi16>) {
func.func @pointwise_mult (%A: memref<2048xi16>, %B: memref<2048xi16>, %C: memref<2048xi16>) {
    affine.for %arg0 = 0 to 2048 {
       %a = affine.load %A[%arg0] : memref<2048xi16>
       %b = affine.load %B[%arg0] : memref<2048xi16>
       //CHECK: %2 = aievec_aie1.mul %0, %1 : vector<16xi16>, vector<16xi16>, vector<16xi48>
       %c = arith.muli %a, %b : i16
       affine.store %c, %C[%arg0] : memref<2048xi16>
    }
    return
}
