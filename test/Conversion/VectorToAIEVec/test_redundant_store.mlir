// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --convert-vector-to-aievec="aie-target=aieml" -canonicalize | FileCheck %s
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
          %10 = affine.load %arg0[%arg3, %arg5 + 2] : memref<64x64xi32>
          %11 = affine.load %arg1[%arg5 + 2, %arg4] : memref<64x64xi32>
          %12 = arith.muli %10, %11 : i32
          %13 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %14 = arith.addi %13, %12 : i32
          affine.store %14, %arg2[%arg3, %arg4] : memref<64x64xi32>
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
//      CHECK:    %[[C2:.*]] = arith.constant 2 : index
//      CHECK:    %[[C16:.*]] = arith.constant 16 : index
//      CHECK:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK:    %[[C64:.*]] = arith.constant 64 : index
//      CHECK:    %[[C1:.*]] = arith.constant 1 : index
//      CHECK:    scf.for %[[A3:.*]] = %[[C0:.*]] to %[[C64:.*]] step %[[C1:.*]] {
//      CHECK:      scf.for %[[A4:.*]] = %[[C0:.*]] to %[[C64:.*]] step %[[C16:.*]] {
//      CHECK:        %[[T0:.*]] = aievec.upd %[[A2]][%[[A3:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:        %[[T1:.*]] = aievec.ups %[[T0:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:        scf.for %[[A5:.*]] = %[[C0:.*]] to %[[C64:.*]] step %[[C16:.*]] {
//      CHECK:          %[[T2:.*]] = aievec.upd %[[A0]][%[[A3:.*]], %[[A5:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T3:.*]] = aievec.broadcast %[[T2:.*]] {idx = 0 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T4:.*]] = aievec.upd %[[A1]][%[[A5:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T5:.*]] = aievec.mac_elem %[[T3:.*]], %[[T4:.*]], %[[T1:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T6:.*]] = aievec.broadcast %[[T2:.*]] {idx = 1 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T7:.*]] = arith.addi %[[A5:.*]], %[[C1:.*]] : index
//      CHECK:          %[[T8:.*]] = aievec.upd %[[A1]][%[[T7:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T9:.*]] = aievec.mac_elem %[[T6:.*]], %[[T8:.*]], %[[T5:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T10:.*]] = aievec.broadcast %[[T2:.*]] {idx = 2 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T11:.*]] = arith.addi %[[A5:.*]], %[[C2:.*]] : index
//      CHECK:          %[[T12:.*]] = aievec.upd %[[A1]][%[[T11:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T13:.*]] = aievec.mac_elem %[[T10:.*]], %[[T12:.*]], %[[T9:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T14:.*]] = aievec.srs %[[T13:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          vector.transfer_write %[[T14:.*]], %[[A2]][%[[A3:.*]], %[[A4:.*]]] {in_bounds = [true]} : vector<16xi32>, memref<64x64xi32>
