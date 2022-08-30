// This test case will directly return the result generated from -affine-super-vectorize
// because in transfer_read %arg1[%arg5, %arg7], lower dim iv %arg7's corresponding loop
// (affine.for %arg7 = #map0(%arg4) to #map1(%arg4))'s upper bound’s affine_map(<(d0) -> (d0 + 4)>)
// result's offset(4) is not divisible by the vector lane size(8).

// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize 2>&1 | FileCheck %s

// CHECK-LABEL: Loop upper bound's affine map offset of inner index of vector.transfer_read is not divisible by number of vector lanes.
// CHECK-LABEL: Cannot apply aie-vectorize to func.func because alignment check has failed.

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 4)>

module {
  func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    affine.for %arg3 = 0 to 64 step 4 {
      affine.for %arg4 = 0 to 64 step 4 {
        affine.for %arg5 = 0 to 64 step 8 {
          affine.for %arg6 = #map0(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map0(%arg4) to #map1(%arg4) {
              %0 = affine.load %arg0[%arg6, %arg5] : memref<64x64xf32>
              %1 = affine.load %arg1[%arg5, %arg7] : memref<64x64xf32>
              %2 = arith.mulf %0, %1 : f32
              %3 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %4 = arith.addf %3, %2 : f32
              affine.store %4, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %6 = affine.load %arg0[%arg6, %arg5+1] : memref<64x64xf32>
              %7 = affine.load %arg1[%arg5+1, %arg7] : memref<64x64xf32>
              %8 = arith.mulf %6, %7 : f32
              %9 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %10 = arith.addf %9, %8 : f32
              affine.store %10, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %12 = affine.load %arg0[%arg6, %arg5+2] : memref<64x64xf32>
              %13 = affine.load %arg1[%arg5+2, %arg7] : memref<64x64xf32>
              %14 = arith.mulf %12, %13 : f32
              %15 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %16 = arith.addf %15, %14 : f32
              affine.store %16, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %18 = affine.load %arg0[%arg6, %arg5+3] : memref<64x64xf32>
              %19 = affine.load %arg1[%arg5+3, %arg7] : memref<64x64xf32>
              %20 = arith.mulf %18, %19 : f32
              %21 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %22 = arith.addf %21, %20 : f32
              affine.store %22, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %24 = affine.load %arg0[%arg6, %arg5+4] : memref<64x64xf32>
              %25 = affine.load %arg1[%arg5+4, %arg7] : memref<64x64xf32>
              %26 = arith.mulf %24, %25 : f32
              %27 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %28 = arith.addf %27, %26 : f32
              affine.store %28, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %30 = affine.load %arg0[%arg6, %arg5+5] : memref<64x64xf32>
              %31 = affine.load %arg1[%arg5+5, %arg7] : memref<64x64xf32>
              %32 = arith.mulf %30, %31 : f32
              %33 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %34 = arith.addf %33, %32 : f32
              affine.store %34, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %36 = affine.load %arg0[%arg6, %arg5+6] : memref<64x64xf32>
              %37 = affine.load %arg1[%arg5+6, %arg7] : memref<64x64xf32>
              %38 = arith.mulf %36, %37 : f32
              %39 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %40 = arith.addf %39, %38 : f32
              affine.store %40, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %42 = affine.load %arg0[%arg6, %arg5+7] : memref<64x64xf32>
%43 = affine.load %arg1[%arg5+7, %arg7] : memref<64x64xf32>
              %44 = arith.mulf %42, %43 : f32
              %45 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %46 = arith.addf %45, %44 : f32
              affine.store %46, %arg2[%arg6, %arg7] : memref<64x64xf32>
            }
          }
        }
      }
    }
    return
  }
}


//CHECK-LABEL:#map0 = affine_map<(d0) -> (d0)>
//CHECK-NEXT: #map1 = affine_map<(d0) -> (d0 + 4)>
//CHECK-NEXT: #map2 = affine_map<(d0, d1) -> (0)>
//CHECK-NEXT: #map3 = affine_map<(d0) -> (d0 + 1)>
//CHECK-NEXT: #map4 = affine_map<(d0) -> (d0 + 2)>
//CHECK-NEXT: #map5 = affine_map<(d0) -> (d0 + 3)>
//CHECK-NEXT: #map6 = affine_map<(d0) -> (d0 + 5)>
//CHECK-NEXT: #map7 = affine_map<(d0) -> (d0 + 6)>
//CHECK-NEXT: #map8 = affine_map<(d0) -> (d0 + 7)>
//CHECK-NEXT: module {
//CHECK-NEXT:   func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
//CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:     affine.for %arg3 = 0 to 64 step 4 {
//CHECK-NEXT:       affine.for %arg4 = 0 to 64 step 4 {
//CHECK-NEXT:         affine.for %arg5 = 0 to 64 step 8 {
//CHECK-NEXT:           affine.for %arg6 = #map0(%arg3) to #map1(%arg3) {
//CHECK-NEXT:             affine.for %arg7 = #map0(%arg4) to #map1(%arg4) step 8 {
//CHECK-NEXT:               %0 = vector.transfer_read %arg0[%arg6, %arg5], %cst {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %1 = vector.transfer_read %arg1[%arg5, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %2 = arith.mulf %0, %1 : vector<8xf32>
//CHECK-NEXT:               %3 = vector.transfer_read %arg2[%arg6, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %4 = arith.addf %3, %2 : vector<8xf32>
//CHECK-NEXT:               %5 = affine.apply #map3(%arg5)
//CHECK-NEXT:               %6 = vector.transfer_read %arg0[%arg6, %5], %cst {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %7 = affine.apply #map3(%arg5)
//CHECK-NEXT:               %8 = vector.transfer_read %arg1[%7, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %9 = arith.mulf %6, %8 : vector<8xf32>
//CHECK-NEXT:               %10 = arith.addf %4, %9 : vector<8xf32>
//CHECK-NEXT:               %11 = affine.apply #map4(%arg5)
//CHECK-NEXT:               %12 = vector.transfer_read %arg0[%arg6, %11], %cst {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %13 = affine.apply #map4(%arg5)
//CHECK-NEXT:               %14 = vector.transfer_read %arg1[%13, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %15 = arith.mulf %12, %14 : vector<8xf32>
//CHECK-NEXT:               %16 = arith.addf %10, %15 : vector<8xf32>
//CHECK-NEXT:               %17 = affine.apply #map5(%arg5)
//CHECK-NEXT:               %18 = vector.transfer_read %arg0[%arg6, %17], %cst {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %19 = affine.apply #map5(%arg5)
//CHECK-NEXT:               %20 = vector.transfer_read %arg1[%19, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %21 = arith.mulf %18, %20 : vector<8xf32>
//CHECK-NEXT:               %22 = arith.addf %16, %21 : vector<8xf32>
//CHECK-NEXT:               %23 = affine.apply #map1(%arg5)
//CHECK-NEXT:               %24 = vector.transfer_read %arg0[%arg6, %23], %cst {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %25 = affine.apply #map1(%arg5)
//CHECK-NEXT:               %26 = vector.transfer_read %arg1[%25, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %27 = arith.mulf %24, %26 : vector<8xf32>
//CHECK-NEXT:               %28 = arith.addf %22, %27 : vector<8xf32>
//CHECK-NEXT:               %29 = affine.apply #map6(%arg5)
//CHECK-NEXT:               %30 = vector.transfer_read %arg0[%arg6, %29], %cst {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %31 = affine.apply #map6(%arg5)
//CHECK-NEXT:               %32 = vector.transfer_read %arg1[%31, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %33 = arith.mulf %30, %32 : vector<8xf32>
//CHECK-NEXT:               %34 = arith.addf %28, %33 : vector<8xf32>
//CHECK-NEXT:               %35 = affine.apply #map7(%arg5)
//CHECK-NEXT:               %36 = vector.transfer_read %arg0[%arg6, %35], %cst {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %37 = affine.apply #map7(%arg5)
//CHECK-NEXT:               %38 = vector.transfer_read %arg1[%37, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %39 = arith.mulf %36, %38 : vector<8xf32>
//CHECK-NEXT:               %40 = arith.addf %34, %39 : vector<8xf32>
//CHECK-NEXT:               %41 = affine.apply #map8(%arg5)
//CHECK-NEXT:               %42 = vector.transfer_read %arg0[%arg6, %41], %cst {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %43 = affine.apply #map8(%arg5)
//CHECK-NEXT:               %44 = vector.transfer_read %arg1[%43, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:               %45 = arith.mulf %42, %44 : vector<8xf32>
//CHECK-NEXT:               %46 = arith.addf %40, %45 : vector<8xf32>
//CHECK-NEXT:               vector.transfer_write %46, %arg2[%arg6, %arg7] {in_bounds = [true]} : vector<8xf32>, memref<64x64xf32>
//CHECK-NEXT:             }
//CHECK-NEXT:           }
//CHECK-NEXT:         }
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     return
//CHECK-NEXT:   }
//CHECK-NEXT: }
