// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 shift=10" | FileCheck %s

func.func @conv2d(%arg0: memref<18x288xi16>, %arg1: memref<9xi16>, %arg2: memref<16x256xi16>) {
  %c0 = arith.constant 0 : index
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  affine.for %arg3 = 0 to 16 {
    affine.for %arg4 = 0 to 256 step 16 {
      %0 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi16>, vector<32xi16>
      %sbh = aievec.ext %0 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
      %sth = aievec.ext %0 {index = 1 : i8} : vector<32xi16>, vector<16xi16>
      %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<9xi16>, vector<16xi16>
      %2 = aievec.broadcast %1 {idx = 0 : i8} : vector<16xi16>, vector<16xi16>
      %3 = arith.muli %sbh, %2 : vector<16xi16>
      %4 = aievec.shift %sbh, %sth, %c2_i32 {isAcc = false} : vector<16xi16>, vector<16xi16>, i32, vector<16xi16>
      %5 = aievec.broadcast %1 {idx = 1 : i8} : vector<16xi16>, vector<16xi16>
      %6 = arith.muli %4, %5 : vector<16xi16>
      %7 = arith.addi %3, %6 : vector<16xi16>
      %8 = aievec.shift %sbh, %sth, %c4_i32 {isAcc = false} : vector<16xi16>, vector<16xi16>, i32, vector<16xi16>
      %9 = aievec.broadcast %1 {idx = 2 : i8} : vector<16xi16>, vector<16xi16>
      %10 = arith.muli %8, %9 : vector<16xi16>
      %11 = arith.addi %7, %10 : vector<16xi16>
      vector.transfer_write %11, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<16xi16>, memref<16x256xi16>
    }
  }
  return
}

// CHECK-LABEL: func @conv2d
//  CHECK-SAME: %[[A0:[A-Za-z0-9]+]]: memref<18x288xi16>
//  CHECK-SAME: %[[A1:[A-Za-z0-9]+]]: memref<9xi16>
//  CHECK-SAME: %[[A2:[A-Za-z0-9]+]]: memref<16x256xi16>
//       CHECK:    %[[C10:.*]] = arith.constant 10 : i32
//       CHECK:    %[[C0:.*]] = arith.constant 0 : index
//       CHECK:    %[[T0:.*]] = aievec.upd %[[A1]][%[[C0]]] {index = 0 : i8, offset = 0 : i32} : memref<9xi16>, vector<16xi16>
//       CHECK:    %[[T1:.*]] = aievec.concat %[[T0]], %[[T0]] : vector<16xi16>, vector<32xi16>
//       CHECK:    affine.for %[[A3:.*]] = 0 to 16 {
//       CHECK:      affine.for %[[A4:.*]] = 0 to 256 step 16 {
//       CHECK:        %[[T2:.*]] = aievec.upd %[[A0]][%[[A3]], %[[A4]]] {index = 0 : i8, offset = 0 : i32} : memref<18x288xi16>, vector<32xi16>
//       CHECK:        %[[T3:.*]] = aievec.mul_conv %[[T2]], %[[T1]] {M = 16 : i32, N = 4 : i32} : vector<32xi16>, vector<32xi16>, vector<16xi64>
//       CHECK:        %[[T4:.*]] = aievec.srs %[[T3]], %[[C10]] : vector<16xi64>, i32, vector<16xi16>
//       CHECK:        vector.transfer_write %[[T4]], %[[A2]][%[[A3]], %[[A4]]] {in_bounds = [true]} : vector<16xi16>, memref<16x256xi16>
