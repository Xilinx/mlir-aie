// This test case will directly return the result generated from -affine-super-vectorize
// because in transfer_read %arg0[%arg3, %arg5], the lowest dim(%arg5)'s corresponding
// loop step(4) is not divisible by the vector lanes(8).

// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize 2>&1 | FileCheck %s

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


// CHECK:       #map = affine_map<(d0, d1) -> (0)>
// CHECK:       #map1 = affine_map<(d0) -> (d0 + 1)>
// CHECK:       #map2 = affine_map<(d0) -> (d0 + 2)>
// CHECK:       #map3 = affine_map<(d0) -> (d0 + 3)>
// CHECK:       module {
// CHECK:         func.func @matmul(%[[VAL_0:.*]]: memref<64x64xf32>, %[[VAL_1:.*]]: memref<64x64xf32>, %[[VAL_2:.*]]: memref<64x64xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           affine.for %[[VAL_4:.*]] = 0 to 64 {
// CHECK:             affine.for %[[VAL_5:.*]] = 0 to 64 step 8 {
// CHECK:               affine.for %[[VAL_6:.*]] = 0 to 64 step 4 {
// CHECK:                 %[[VAL_7:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_4]], %[[VAL_6]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
// CHECK:                 %[[VAL_8:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_6]], %[[VAL_5]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                 %[[VAL_9:.*]] = arith.mulf %[[VAL_7]], %[[VAL_8]] : vector<8xf32>
// CHECK:                 %[[VAL_10:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_5]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                 %[[VAL_11:.*]] = arith.addf %[[VAL_10]], %[[VAL_9]] : vector<8xf32>
// CHECK:                 %[[VAL_12:.*]] = affine.apply #map1(%[[VAL_6]])
// CHECK:                 %[[VAL_13:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_4]], %[[VAL_12]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
// CHECK:                 %[[VAL_14:.*]] = affine.apply #map1(%[[VAL_6]])
// CHECK:                 %[[VAL_15:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_14]], %[[VAL_5]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                 %[[VAL_16:.*]] = arith.mulf %[[VAL_13]], %[[VAL_15]] : vector<8xf32>
// CHECK:                 %[[VAL_17:.*]] = arith.addf %[[VAL_11]], %[[VAL_16]] : vector<8xf32>
// CHECK:                 %[[VAL_18:.*]] = affine.apply #map2(%[[VAL_6]])
// CHECK:                 %[[VAL_19:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_4]], %[[VAL_18]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
// CHECK:                 %[[VAL_20:.*]] = affine.apply #map2(%[[VAL_6]])
// CHECK:                 %[[VAL_21:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_20]], %[[VAL_5]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                 %[[VAL_22:.*]] = arith.mulf %[[VAL_19]], %[[VAL_21]] : vector<8xf32>
// CHECK:                 %[[VAL_23:.*]] = arith.addf %[[VAL_17]], %[[VAL_22]] : vector<8xf32>
// CHECK:                 %[[VAL_24:.*]] = affine.apply #map3(%[[VAL_6]])
// CHECK:                 %[[VAL_25:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_4]], %[[VAL_24]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
// CHECK:                 %[[VAL_26:.*]] = affine.apply #map3(%[[VAL_6]])
// CHECK:                 %[[VAL_27:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_26]], %[[VAL_5]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                 %[[VAL_28:.*]] = arith.mulf %[[VAL_25]], %[[VAL_27]] : vector<8xf32>
// CHECK:                 %[[VAL_29:.*]] = arith.addf %[[VAL_23]], %[[VAL_28]] : vector<8xf32>
// CHECK:                 vector.transfer_write %[[VAL_29]], %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_5]]] {in_bounds = [true]} : vector<8xf32>, memref<64x64xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }