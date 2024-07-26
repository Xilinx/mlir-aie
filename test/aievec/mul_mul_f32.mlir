// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize="shift=10" -split-input-file | FileCheck %s
// XFAIL: *

func @mul_mul (%A: memref<2048xf32>, %B: memref<2048xf32>, %C: memref<2048xf32>, %cc: f32) {
// CHECK-LABEL: func @mul_mul
// CHECK:  %2 = aievec.upd %arg0[%arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048xf32>, vector<8xf32>
// CHECK:  %3 = aievec.upd %arg1[%arg4] {index = 0 : i8, offset = 0 : i32} : memref<2048xf32>, vector<8xf32>
// CHECK:  %4 = aievec_aie1.mul %1, %2 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x76543210", zstart = "0"} : vector<16xf32>, vector<8xf32>, !aievec.acc<8xf32>
// CHECK:  %5 = aievec.srs %4 {shift = 10 : i8} : !aievec.acc<8xf32>, vector<8xf32>
// CHECK:  %6 = arith.mulf %5, %3 : vector<8xf32>
// CHECK:  vector.transfer_write %6, %arg2[%arg4] {in_bounds = [true]} : vector<8xf32>, memref<2048xf32>
    affine.for %arg0 = 0 to 2048 {
       %a = affine.load %A[%arg0] : memref<2048xf32>
       %b = affine.load %B[%arg0] : memref<2048xf32>
       %t = arith.mulf %cc, %a : f32
       %c = arith.mulf %t, %b : f32
       affine.store %c, %C[%arg0] : memref<2048xf32>
    }
    return
}
