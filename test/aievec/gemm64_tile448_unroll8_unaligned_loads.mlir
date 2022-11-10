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

// CHECK:       #map = affine_map<(d0) -> (d0)>
// CHECK:       #map1 = affine_map<(d0) -> (d0 + 4)>
// CHECK:       #map2 = affine_map<(d0, d1) -> (0)>
// CHECK:       #map3 = affine_map<(d0) -> (d0 + 1)>
// CHECK:       #map4 = affine_map<(d0) -> (d0 + 2)>
// CHECK:       #map5 = affine_map<(d0) -> (d0 + 3)>
// CHECK:       #map6 = affine_map<(d0) -> (d0 + 5)>
// CHECK:       #map7 = affine_map<(d0) -> (d0 + 6)>
// CHECK:       #map8 = affine_map<(d0) -> (d0 + 7)>
// CHECK:       module {
// CHECK:         func.func @matmul(%[[VAL_0:.*]]: memref<64x64xf32>, %[[VAL_1:.*]]: memref<64x64xf32>, %[[VAL_2:.*]]: memref<64x64xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           affine.for %[[VAL_4:.*]] = 0 to 64 step 4 {
// CHECK:             affine.for %[[VAL_5:.*]] = 0 to 64 step 4 {
// CHECK:               affine.for %[[VAL_6:.*]] = 0 to 64 step 8 {
// CHECK:                 affine.for %[[VAL_7:.*]] = #map(%[[VAL_4]]) to #map1(%[[VAL_4]]) {
// CHECK:                   affine.for %[[VAL_8:.*]] = #map(%[[VAL_5]]) to #map1(%[[VAL_5]]) step 8 {
// CHECK:                     %[[VAL_9:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_6]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_10:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_6]], %[[VAL_8]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_11:.*]] = arith.mulf %[[VAL_9]], %[[VAL_10]] : vector<8xf32>
// CHECK:                     %[[VAL_12:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_7]], %[[VAL_8]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_13:.*]] = arith.addf %[[VAL_12]], %[[VAL_11]] : vector<8xf32>
// CHECK:                     %[[VAL_14:.*]] = affine.apply #map3(%[[VAL_6]])
// CHECK:                     %[[VAL_15:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_14]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_16:.*]] = affine.apply #map3(%[[VAL_6]])
// CHECK:                     %[[VAL_17:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_16]], %[[VAL_8]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_18:.*]] = arith.mulf %[[VAL_15]], %[[VAL_17]] : vector<8xf32>
// CHECK:                     %[[VAL_19:.*]] = arith.addf %[[VAL_13]], %[[VAL_18]] : vector<8xf32>
// CHECK:                     %[[VAL_20:.*]] = affine.apply #map4(%[[VAL_6]])
// CHECK:                     %[[VAL_21:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_20]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_22:.*]] = affine.apply #map4(%[[VAL_6]])
// CHECK:                     %[[VAL_23:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_22]], %[[VAL_8]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_24:.*]] = arith.mulf %[[VAL_21]], %[[VAL_23]] : vector<8xf32>
// CHECK:                     %[[VAL_25:.*]] = arith.addf %[[VAL_19]], %[[VAL_24]] : vector<8xf32>
// CHECK:                     %[[VAL_26:.*]] = affine.apply #map5(%[[VAL_6]])
// CHECK:                     %[[VAL_27:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_26]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_28:.*]] = affine.apply #map5(%[[VAL_6]])
// CHECK:                     %[[VAL_29:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_28]], %[[VAL_8]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_30:.*]] = arith.mulf %[[VAL_27]], %[[VAL_29]] : vector<8xf32>
// CHECK:                     %[[VAL_31:.*]] = arith.addf %[[VAL_25]], %[[VAL_30]] : vector<8xf32>
// CHECK:                     %[[VAL_32:.*]] = affine.apply #map1(%[[VAL_6]])
// CHECK:                     %[[VAL_33:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_32]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_34:.*]] = affine.apply #map1(%[[VAL_6]])
// CHECK:                     %[[VAL_35:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_34]], %[[VAL_8]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_36:.*]] = arith.mulf %[[VAL_33]], %[[VAL_35]] : vector<8xf32>
// CHECK:                     %[[VAL_37:.*]] = arith.addf %[[VAL_31]], %[[VAL_36]] : vector<8xf32>
// CHECK:                     %[[VAL_38:.*]] = affine.apply #map6(%[[VAL_6]])
// CHECK:                     %[[VAL_39:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_38]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_40:.*]] = affine.apply #map6(%[[VAL_6]])
// CHECK:                     %[[VAL_41:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_40]], %[[VAL_8]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_42:.*]] = arith.mulf %[[VAL_39]], %[[VAL_41]] : vector<8xf32>
// CHECK:                     %[[VAL_43:.*]] = arith.addf %[[VAL_37]], %[[VAL_42]] : vector<8xf32>
// CHECK:                     %[[VAL_44:.*]] = affine.apply #map7(%[[VAL_6]])
// CHECK:                     %[[VAL_45:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_44]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_46:.*]] = affine.apply #map7(%[[VAL_6]])
// CHECK:                     %[[VAL_47:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_46]], %[[VAL_8]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_48:.*]] = arith.mulf %[[VAL_45]], %[[VAL_47]] : vector<8xf32>
// CHECK:                     %[[VAL_49:.*]] = arith.addf %[[VAL_43]], %[[VAL_48]] : vector<8xf32>
// CHECK:                     %[[VAL_50:.*]] = affine.apply #map8(%[[VAL_6]])
// CHECK:                     %[[VAL_51:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_50]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #map2} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_52:.*]] = affine.apply #map8(%[[VAL_6]])
// CHECK:                     %[[VAL_53:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_52]], %[[VAL_8]]], %[[VAL_3]] {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
// CHECK:                     %[[VAL_54:.*]] = arith.mulf %[[VAL_51]], %[[VAL_53]] : vector<8xf32>
// CHECK:                     %[[VAL_55:.*]] = arith.addf %[[VAL_49]], %[[VAL_54]] : vector<8xf32>
// CHECK:                     vector.transfer_write %[[VAL_55]], %[[VAL_2]]{{\[}}%[[VAL_7]], %[[VAL_8]]] {in_bounds = [true]} : vector<8xf32>, memref<64x64xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }