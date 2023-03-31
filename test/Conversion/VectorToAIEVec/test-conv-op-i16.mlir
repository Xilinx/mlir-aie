// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --convert-vector-to-aievec="aie-target=aieml shift=10" | FileCheck %s

func.func @conv2d(%arg0: memref<18x288xi16>, %arg1: memref<9xi16>, %arg2: memref<16x256xi16>) {
  %c0 = arith.constant 0 : index
  affine.for %arg3 = 0 to 16 {
    affine.for %arg4 = 0 to 256 step 16 {
      %0 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi16>, vector<16xi16>
      %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<9xi16>, vector<16xi16>
      %2 = aievec.broadcast %1 {idx = 0 : i8} : vector<16xi16>, vector<16xi16>
      %3 = arith.muli %0, %2 : vector<16xi16>
      %4 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg4)
      %5 = aievec.upd %arg0[%arg3, %4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi16>, vector<16xi16>
      %6 = aievec.broadcast %1 {idx = 1 : i8} : vector<16xi16>, vector<16xi16>
      %7 = arith.muli %5, %6 : vector<16xi16>
      %8 = arith.addi %3, %7 : vector<16xi16>
      %9 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg4)
      %10 = aievec.upd %arg0[%arg3, %9] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi16>, vector<16xi16>
      %11 = aievec.broadcast %1 {idx = 2 : i8} : vector<16xi16>, vector<16xi16>
      %12 = arith.muli %10, %11 : vector<16xi16>
      %13 = arith.addi %8, %12 : vector<16xi16>
      vector.transfer_write %13, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<16xi16>, memref<16x256xi16>
    }
  }
  return
}

// CHECK-LABEL:  func @conv2d
// CHECK-SAME: %[[A0:[A-Za-z0-9]+]]: memref<18x288xi16>
// CHECK-SAME: %[[A1:[A-Za-z0-9]+]]: memref<9xi16>
// CHECK-SAME: %[[A2:[A-Za-z0-9]+]]: memref<16x256xi16>
//      CHECK:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK:    %[[T0:.*]] = aievec.upd %[[A1:.*]][%[[C0:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<9xi16>, vector<32xi16>
//      CHECK:    affine.for %[[A3:.*]] = 0 to 16 {
//      CHECK:      affine.for %[[A4:.*]] = 0 to 256 step 16 {
//      CHECK:        %[[T1:.*]] = aievec.upd %[[A0:.*]][%[[A3:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi16>, vector<32xi16>
//      CHECK:        %[[T2:.*]] = aievec.mul_conv %[[T1:.*]], %[[T0:.*]] {M = 16 : i32, N = 4 : i32} : vector<32xi16>, vector<32xi16>, vector<16xi64>
//      CHECK:        %[[T3:.*]] = aievec.srs %[[T2:.*]] {shift = 10 : i8} : vector<16xi64>, vector<16xi16>
//      CHECK:        vector.transfer_write %[[T3:.*]], %[[A2:.*]][%[[A3:.*]], %[[A4:.*]]] {in_bounds = [true]} : vector<16xi16>, memref<16x256xi16>
