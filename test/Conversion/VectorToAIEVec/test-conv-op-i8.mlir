// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s

func.func @conv2d(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
  %c0 = arith.constant 0 : index
  affine.for %arg3 = 0 to 16 {
    affine.for %arg4 = 0 to 256 step 32 {
      %0 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<32xi8>
      %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<48xi8>, vector<32xi8>
      %2 = aievec.broadcast %1 {idx = 0 : i8} : vector<32xi8>, vector<32xi8>
      %3 = arith.muli %0, %2 : vector<32xi8>
      %4 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg4)
      %5 = aievec.upd %arg0[%arg3, %4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<32xi8>
      %6 = aievec.broadcast %1 {idx = 2 : i8} : vector<32xi8>, vector<32xi8>
      %7 = arith.muli %5, %6 : vector<32xi8>
      %8 = arith.addi %3, %7 : vector<32xi8>
      %9 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg4)
      %10 = aievec.upd %arg0[%arg3, %9] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<32xi8>
      %11 = aievec.broadcast %1 {idx = 4 : i8} : vector<32xi8>, vector<32xi8>
      %12 = arith.muli %10, %11 : vector<32xi8>
      %13 = arith.addi %8, %12 : vector<32xi8>
      vector.transfer_write %13, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<32xi8>, memref<16x256xi8>
    }
  }
  return
}

// CHECK-LABEL:  func @conv2d
// CHECK-SAME: %[[A0:[A-Za-z0-9]+]]: memref<18x288xi8>
// CHECK-SAME: %[[A1:[A-Za-z0-9]+]]: memref<48xi8>
// CHECK-SAME: %[[A2:[A-Za-z0-9]+]]: memref<16x256xi8>
//      CHECK:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK:    %[[T0:.*]] = aievec.upd %[[A1:.*]][%[[C0:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<48xi8>, vector<64xi8>
//      CHECK:    %[[T1:.*]] = aievec.shuffle %[[T0:.*]] {mode = 0 : i32} : vector<64xi8>, vector<64xi8>
//      CHECK:    affine.for %[[A3:.*]] = 0 to 16 {
//      CHECK:      affine.for %[[A4:.*]] = 0 to 256 step 32 {
//      CHECK:        %[[T2:.*]] = aievec.upd %[[A0:.*]][%[[A3:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<64xi8>
//      CHECK:        %[[T3:.*]] = aievec.mul_conv %[[T2:.*]], %[[T1:.*]] {M = 32 : i32, N = 8 : i32} : vector<64xi8>, vector<64xi8>, vector<32xi32>
//      CHECK:        %[[T4:.*]] = aievec.srs %[[T3:.*]] {shift = 0 : i8} : vector<32xi32>, vector<32xi8>
//      CHECK:        vector.transfer_write %[[T4:.*]], %[[A2:.*]][%[[A3:.*]], %[[A4:.*]]] {in_bounds = [true]} : vector<32xi8>, memref<16x256xi8>
