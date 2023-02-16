// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" -redundant-load-store-optimization -transfer-op-flow-opt -canonicalize | FileCheck %s
module {
  func.func @matmul(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        affine.for %arg5 = 0 to 64 step 16 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<64x64xi32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<64x64xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %5 = affine.load %arg0[%arg3, %arg5 + 1] : memref<64x64xi32>
          %6 = affine.load %arg1[%arg5 + 1, %arg4] : memref<64x64xi32>
          %7 = arith.muli %5, %6 : i32
          %8 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %9 = arith.addi %8, %7 : i32
          affine.store %9, %arg2[%arg3, %arg4] : memref<64x64xi32>
        }
      }
    }
    return
  }
}

// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A0:[0-9a-zA-Z]*]]: memref<64x64xi32>
// CHECK-SAME: %[[A1:[0-9a-zA-Z]*]]: memref<64x64xi32>
// CHECK-SAME: %[[A2:[0-9a-zA-Z]*]]: memref<64x64xi32>
//      CHECK:    %[[C0:.*]] = arith.constant 0 : i32
//      CHECK:    affine.for %[[A3:.*]] = 0 to 64 {
//      CHECK:      affine.for %[[A4:.*]] = 0 to 64 step 16 {
//      CHECK:        affine.for %[[A5:.*]] = 0 to 64 step 16 {
//      CHECK:          %[[T0:.*]] = vector.transfer_read %[[A0]][%[[A3:.*]], %[[A5:.*]]], %[[C0:.*]] {in_bounds = [true], permutation_map = #map} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T1:.*]] = vector.transfer_read %[[A1]][%[[A5:.*]], %[[A4:.*]]], %[[C0:.*]] {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T2:.*]] = arith.muli %[[T0:.*]], %[[T1:.*]] : vector<16xi32>
//      CHECK:          %[[T3:.*]] = vector.transfer_read %[[A2]][%[[A3:.*]], %[[A4:.*]]], %[[C0:.*]] {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T4:.*]] = arith.addi %[[T3:.*]], %[[T2:.*]] : vector<16xi32>
//      CHECK:          %[[T5:.*]] = affine.apply #map1(%[[A5:.*]])
//      CHECK:          %[[T6:.*]] = vector.transfer_read %[[A0]][%[[A3:.*]], %[[T5:.*]]], %[[C0:.*]] {in_bounds = [true], permutation_map = #map} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T7:.*]] = affine.apply #map1(%[[A5:.*]])
//      CHECK:          %[[T8:.*]] = vector.transfer_read %[[A1]][%[[T7:.*]], %[[A4:.*]]], %[[C0:.*]] {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T9:.*]] = arith.muli %[[T6:.*]], %[[T8:.*]] : vector<16xi32>
//      CHECK:          %[[T10:.*]] = arith.addi %[[T4:.*]], %[[T9:.*]] : vector<16xi32>
//      CHECK:          vector.transfer_write %[[T10:.*]], %[[A2]][%[[A3:.*]], %[[A4:.*]]] {in_bounds = [true]} : vector<16xi32>, memref<64x64xi32>
