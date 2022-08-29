// This test case will directly return the result generated from -affine-super-vectorize
// because in transfer_read %arg0[%arg3, %arg5], the lowest dim(%arg5)'s corresponding
// loop step(4) is not divisible by the vector lanes(8).

// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize -unaligned-loads-check=true --debug-only=aie-vect -split-input-file 2>&1 | FileCheck %s

//CHECK-LABEL: %0 = vector.transfer_read %arg0[%arg3, %arg5], %cst {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xf32>, vector<8xf32>'s lowest dim's loop step is not divisible by the vector lanes.
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


//CHECK-LABEL:#map0 = affine_map<(d0, d1) -> (0)>
//CHECK-NEXT: #map1 = affine_map<(d0) -> (d0 + 1)>
//CHECK-NEXT: #map2 = affine_map<(d0) -> (d0 + 2)>
//CHECK-NEXT: #map3 = affine_map<(d0) -> (d0 + 3)>
//CHECK-NEXT: module {
//CHECK-NEXT:   func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
//CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:     affine.for %arg3 = 0 to 64 {
//CHECK-NEXT:       affine.for %arg4 = 0 to 64 step 8 {
//CHECK-NEXT:         affine.for %arg5 = 0 to 64 step 4 {
//CHECK-NEXT:           %0 = vector.transfer_read %arg0[%arg3, %arg5], %cst {in_bounds = [true], permutation_map = #map0} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:           %1 = vector.transfer_read %arg1[%arg5, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:           %2 = arith.mulf %0, %1 : vector<8xf32>
//CHECK-NEXT:           %3 = vector.transfer_read %arg2[%arg3, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:           %4 = arith.addf %3, %2 : vector<8xf32>
//CHECK-NEXT:           %5 = affine.apply #map1(%arg5)
//CHECK-NEXT:           %6 = vector.transfer_read %arg0[%arg3, %5], %cst {in_bounds = [true], permutation_map = #map0} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:           %7 = affine.apply #map1(%arg5)
//CHECK-NEXT:           %8 = vector.transfer_read %arg1[%7, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:           %9 = arith.mulf %6, %8 : vector<8xf32>
//CHECK-NEXT:           %10 = arith.addf %4, %9 : vector<8xf32>
//CHECK-NEXT:           %11 = affine.apply #map2(%arg5)
//CHECK-NEXT:           %12 = vector.transfer_read %arg0[%arg3, %11], %cst {in_bounds = [true], permutation_map = #map0} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:           %13 = affine.apply #map2(%arg5)
//CHECK-NEXT:           %14 = vector.transfer_read %arg1[%13, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:           %15 = arith.mulf %12, %14 : vector<8xf32>
//CHECK-NEXT:           %16 = arith.addf %10, %15 : vector<8xf32>
//CHECK-NEXT:           %17 = affine.apply #map3(%arg5)
//CHECK-NEXT:           %18 = vector.transfer_read %arg0[%arg3, %17], %cst {in_bounds = [true], permutation_map = #map0} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:           %19 = affine.apply #map3(%arg5)
//CHECK-NEXT:           %20 = vector.transfer_read %arg1[%19, %arg4], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:           %21 = arith.mulf %18, %20 : vector<8xf32>
//CHECK-NEXT:           %22 = arith.addf %16, %21 : vector<8xf32>
//CHECK-NEXT:           vector.transfer_write %22, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xf32>, memref<64x64xf32>
//CHECK-NEXT:         }
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     return
//CHECK-NEXT:   }
//CHECK-NEXT: }
