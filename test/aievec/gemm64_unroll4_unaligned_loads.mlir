// This test case will directly return the result generated from -affine-super-vectorize
// because in transfer_read %arg0[%arg3, %arg5], the lowest dim(%arg5)'s corresponding
// loop step(4) is not divisible by the vector lanes(8).

// RUN: aie-opt %s --aie-affine-vectorize="virtual-vector-size=8" 2>&1 | FileCheck %s

// CHECK-LABEL: Loop step of inner index of vector.transfer_read is not divisible by number of vector lanes.
// CHECK-LABEL: Cannot apply aie-vectorize to func.func because alignment check has failed.

module {
  func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        affine.for %arg5 = 0 to 64 step 4 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<64x64xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<64x64xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = affine.load %arg2[%arg3, %arg4] : memref<64x64xf32>
          %4 = arith.addf %3, %2 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<64x64xf32>
          %5 = affine.load %arg0[%arg3, %arg5 + 1] : memref<64x64xf32>
          %6 = affine.load %arg1[%arg5 + 1, %arg4] : memref<64x64xf32>
          %7 = arith.mulf %5, %6 : f32
          %8 = affine.load %arg2[%arg3, %arg4] : memref<64x64xf32>
          %9 = arith.addf %8, %7 : f32
          affine.store %9, %arg2[%arg3, %arg4] : memref<64x64xf32>
          %10 = affine.load %arg0[%arg3, %arg5 + 2] : memref<64x64xf32>
          %11 = affine.load %arg1[%arg5 + 2, %arg4] : memref<64x64xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = affine.load %arg2[%arg3, %arg4] : memref<64x64xf32>
          %14 = arith.addf %13, %12 : f32
          affine.store %14, %arg2[%arg3, %arg4] : memref<64x64xf32>
          %15 = affine.load %arg0[%arg3, %arg5 + 3] : memref<64x64xf32>
          %16 = affine.load %arg1[%arg5 + 3, %arg4] : memref<64x64xf32>
          %17 = arith.mulf %15, %16 : f32
          %18 = affine.load %arg2[%arg3, %arg4] : memref<64x64xf32>
          %19 = arith.addf %18, %17 : f32
          affine.store %19, %arg2[%arg3, %arg4] : memref<64x64xf32>
        }
      }
    }
    return
  }
}

//CHECK-LABEL:#map = affine_map<(d0, d1) -> (0)>
//CHECK-NEXT: module {
//CHECK-NEXT:  func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
//CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %c64 = arith.constant 64 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    scf.for %arg3 = %c0 to %c64 step %c1 {
//CHECK-NEXT:      %c0_0 = arith.constant 0 : index
//CHECK-NEXT:      %c64_1 = arith.constant 64 : index
//CHECK-NEXT:      %c8 = arith.constant 8 : index
//CHECK-NEXT:      scf.for %arg4 = %c0_0 to %c64_1 step %c8 {
//CHECK-NEXT:        %c0_2 = arith.constant 0 : index
//CHECK-NEXT:        %c64_3 = arith.constant 64 : index
//CHECK-NEXT:        %c4 = arith.constant 4 : index
//CHECK-NEXT:        scf.for %arg5 = %c0_2 to %c64_3 step %c4 {
//CHECK-NEXT:          %0 = vector.transfer_read %arg0[%arg3, %arg5], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:          %1 = vector.transfer_read %arg1[%arg5, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:          %2 = arith.mulf %0, %1 : vector<8xf32>
//CHECK-NEXT:          %3 = vector.transfer_read %arg2[%arg3, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:          %4 = arith.addf %3, %2 : vector<8xf32>
//CHECK-NEXT:          %c1_4 = arith.constant 1 : index
//CHECK-NEXT:          %5 = arith.addi %arg5, %c1_4 : index
//CHECK-NEXT:          %6 = vector.transfer_read %arg0[%arg3, %5], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:          %7 = vector.transfer_read %arg1[%5, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:          %8 = arith.mulf %6, %7 : vector<8xf32>
//CHECK-NEXT:          %9 = arith.addf %4, %8 : vector<8xf32>
//CHECK-NEXT:          %c2 = arith.constant 2 : index
//CHECK-NEXT:          %10 = arith.addi %arg5, %c2 : index
//CHECK-NEXT:          %11 = vector.transfer_read %arg0[%arg3, %10], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:          %12 = vector.transfer_read %arg1[%10, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:          %13 = arith.mulf %11, %12 : vector<8xf32>
//CHECK-NEXT:          %14 = arith.addf %9, %13 : vector<8xf32>
//CHECK-NEXT:          %c3 = arith.constant 3 : index
//CHECK-NEXT:          %15 = arith.addi %arg5, %c3 : index
//CHECK-NEXT:          %16 = vector.transfer_read %arg0[%arg3, %15], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:          %17 = vector.transfer_read %arg1[%15, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:          %18 = arith.mulf %16, %17 : vector<8xf32>
//CHECK-NEXT:          %19 = arith.addf %14, %18 : vector<8xf32>
//CHECK-NEXT:          vector.transfer_write %19, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xf32>, memref<64x64xf32>
//CHECK-NEXT:        }
//CHECK-NEXT:      }
//CHECK-NEXT:    }
//CHECK-NEXT:    return
//CHECK-NEXT:  }
//CHECK-NEXT:}
