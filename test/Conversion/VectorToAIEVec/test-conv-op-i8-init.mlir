// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s

func.func @conv2d(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  affine.for %arg3 = 0 to 16 {
    affine.for %arg4 = 0 to 256 step 32 {
      %0 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<16x256xi8>, vector<32xi8>
      %1 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<64xi8>
      %sbh = aievec.ext %1 {index = 0 : i8} : vector<64xi8>, vector<32xi8>
      %sth = aievec.ext %1 {index = 1 : i8} : vector<64xi8>, vector<32xi8>
      %2 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<48xi8>, vector<32xi8>
      %3 = aievec.broadcast %2 {idx = 0 : i8} : vector<32xi8>, vector<32xi8>
      %4 = arith.muli %sbh, %3 : vector<32xi8>
      %5 = arith.addi %0, %4 : vector<32xi8>
      %7 = aievec.shift %sbh, %sth, %c1_i32 {isAcc = false} : vector<32xi8>, vector<32xi8>, i32, vector<32xi8>
      %8 = aievec.broadcast %2 {idx = 2 : i8} : vector<32xi8>, vector<32xi8>
      %9 = arith.muli %7, %8 : vector<32xi8>
      %10 = arith.addi %5, %9 : vector<32xi8>
      %12 = aievec.shift %sbh, %sth, %c2_i32 {isAcc = false} : vector<32xi8>, vector<32xi8>, i32, vector<32xi8>
      %13 = aievec.broadcast %2 {idx = 4 : i8} : vector<32xi8>, vector<32xi8>
      %14 = arith.muli %12, %13 : vector<32xi8>
      %15 = arith.addi %10, %14 : vector<32xi8>
      vector.transfer_write %15, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<32xi8>, memref<16x256xi8>
    }
  }
  return
}

// CHECK-LABEL: func @conv2d
//  CHECK-SAME: %[[A0:[A-Za-z0-9]+]]: memref<18x288xi8>
//  CHECK-SAME: %[[A1:[A-Za-z0-9]+]]: memref<48xi8>
//  CHECK-SAME: %[[A2:[A-Za-z0-9]+]]: memref<16x256xi8>
//       CHECK:    %[[C0I32:.*]] = arith.constant 0 : i32
//       CHECK:    %[[C0:.*]] = arith.constant 0 : index
//       CHECK:    %[[T0:.*]] = aievec.upd %[[A1]][%[[C0]]] {index = 0 : i8, offset = 0 : i32} : memref<48xi8>, vector<32xi8>
//       CHECK:    %[[T1:.*]] = aievec.concat %[[T0]], %[[T0]] : vector<32xi8>, vector<64xi8>
//       CHECK:    %[[T2:.*]] = aievec.shuffle %[[T1]], %[[T1]] [t8_64x2_lo] : vector<64xi8>
//       CHECK:    affine.for %[[I:.*]] = 0 to 16 {
//       CHECK:      affine.for %[[J:.*]] = 0 to 256 step 32 {
//       CHECK:        %[[T3:.*]] = aievec.upd %[[A2]][%[[I]], %[[J]]] {index = 0 : i8, offset = 0 : i32} : memref<16x256xi8>, vector<32xi8>
//       CHECK:        %[[T4:.*]] = aievec.upd %[[A0]][%[[I]], %[[J]]] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi8>, vector<64xi8>
//       CHECK:        %[[T5:.*]] = aievec.ups %[[T3]] {shift = 0 : i8} : vector<32xi8>, vector<32xi32>
//       CHECK:        %[[T6:.*]] = aievec.fma_conv %[[T4]], %[[T2]], %[[T5]] {M = 32 : i32, N = 8 : i32} : vector<64xi8>, vector<64xi8>, vector<32xi32>
//       CHECK:        %[[T7:.*]] = aievec.srs %[[T6]], %[[C0I32]] : vector<32xi32>, i32, vector<32xi8>
//       CHECK:        vector.transfer_write %[[T7]], %[[A2]][%[[I]], %[[J]]] {in_bounds = [true]} : vector<32xi8>, memref<16x256xi8>
