// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s

module {
  func.func @matmul(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 step 16 {
        affine.for %arg5 = 0 to 64 step 16 {
          %0 = vector.transfer_read %arg0[%arg3, %arg5], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %1 = vector.transfer_read %arg1[%arg5, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %2 = arith.muli %0, %1 : vector<16xi32>
          %3 = vector.transfer_read %arg2[%arg3, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %4 = arith.addi %3, %2 : vector<16xi32>
          %5 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg5)
          %6 = vector.transfer_read %arg0[%arg3, %5], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %7 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg5)
          %8 = vector.transfer_read %arg1[%7, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %9 = arith.muli %6, %8 : vector<16xi32>
          %10 = arith.addi %4, %9 : vector<16xi32>
          vector.transfer_write %10, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<16xi32>, memref<64x64xi32>
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
//      CHECK:    affine.for %[[A3:.*]] = 0 to 64 {
//      CHECK:      affine.for %[[A4:.*]] = 0 to 64 step 16 {
//      CHECK:        affine.for %[[A5:.*]] = 0 to 64 step 16 {
//      CHECK:          %[[T0:.*]] = aievec.upd %[[A0]][%[[A3:.*]], %[[A5:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T1:.*]] = aievec.broadcast %[[T0:.*]] {idx = 0 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T2:.*]] = aievec.upd %[[A1]][%[[A5:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T3:.*]] = aievec.upd %[[A2]][%[[A3:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T4:.*]] = aievec.ups %[[T3:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T5:.*]] = aievec.mac_elem %[[T2:.*]], %[[T1:.*]], %[[T4:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T7:.*]] = aievec.broadcast %[[T0:.*]] {idx = 1 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T8:.*]] = affine.apply #map(%[[A5:.*]])
//      CHECK:          %[[T9:.*]] = aievec.upd %[[A1]][%[[T8:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T11:.*]] = aievec.mac_elem %[[T9:.*]], %[[T7:.*]], %[[T5:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T12:.*]] = aievec.srs %[[T11:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          vector.transfer_write %[[T12:.*]], %[[A2]][%[[A3:.*]], %[[A4:.*]]] {in_bounds = [true]} : vector<16xi32>, memref<64x64xi32>

