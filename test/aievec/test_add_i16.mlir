// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" --aie-vectorize -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv2d(%arg0: memref<256xi16>, %arg1: memref<1xi16>, %arg2: memref<256xi16>) {
func.func @conv2d (%A: memref<256xi16>, %B: memref<1xi16>, %C: memref<256xi16>) {
    affine.for %arg0 = 0 to 256 {
      %a1 = affine.load %A[%arg0] : memref<256xi16>
      %b1 = affine.load %B[0] : memref<1xi16>
      //CHECK: %4 = aievec_aie1.add %3, %1 {xoffsets = "0x03020100", xoffsets_hi = "0x07060504", xsquare = "0x3210", xstart = "0", zoffsets = "0", zoffsets_hi = "0", zsquare = "0", zstart = "0"} : vector<32xi16>, vector<32xi16>, vector<32xi16>
      %d1 = arith.addi %a1, %b1 : i16 
      affine.store %d1, %C[%arg0] : memref<256xi16>
    }
    return
}
