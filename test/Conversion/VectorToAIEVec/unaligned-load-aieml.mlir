// RUN: aie-opt %s --convert-vector-to-aievec -split-input-file | FileCheck %s --check-prefix=CHECK

aie.device(npu1) {
   // CHECK-LABEL: func @unaligned_read
   //   CHECK-DAG:    %[[C2i32:.*]] = arith.constant 2 : i32
   //   CHECK-DAG:    %[[C32:.*]] = arith.constant 32 : index
   //   CHECK-DAG:    %[[C16i32:.*]] = arith.constant 16 : i32
   //   CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
   //       CHECK:    %[[T0:.*]] = aievec.upd {{.*}}[%[[C0:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<48xi8>, vector<64xi8>
   //       CHECK:    %[[T0E0:.*]] = aievec.ext %[[T0]] {index = 0 : i8} : vector<64xi8>, vector<32xi8>
   //       CHECK:    %[[T0E1:.*]] = aievec.ext %[[T0]] {index = 1 : i8} : vector<64xi8>, vector<32xi8>
   //       CHECK:    %[[R0:.*]] = aievec.shift %[[T0E0]], %[[T0E1]], %[[C16i32]] {isAcc = false} : vector<32xi8>, vector<32xi8>, i32, vector<32xi8>
   //       CHECK:    %[[T1:.*]] = aievec.upd {{.*}}[%[[C32:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<48xi8>, vector<64xi8>
   //       CHECK:    %[[T1E0:.*]] = aievec.ext %[[T1]] {index = 0 : i8} : vector<64xi8>, vector<32xi8>
   //       CHECK:    %[[T1E1:.*]] = aievec.ext %[[T1]] {index = 1 : i8} : vector<64xi8>, vector<32xi8>
   //       CHECK:    %[[R1:.*]] = aievec.shift %[[T1E0]], %[[T1E1]], %[[C2i32]] {isAcc = false} : vector<32xi8>, vector<32xi8>, i32, vector<32xi8>
   //       CHECK:    return %[[R0:.*]], %[[R1:.*]] : vector<32xi8>, vector<32xi8>
   func.func @unaligned_read(%a: memref<48xi8>) -> (vector<32xi8>, vector<32xi8>) {
      %c0_i8 = arith.constant 0 : i8
      %c16 = arith.constant 16 : index
      %c34 = arith.constant 34 : index
      %0 = vector.transfer_read %a[%c16], %c0_i8 : memref<48xi8>, vector<32xi8>
      %1 = vector.transfer_read %a[%c34], %c0_i8 : memref<48xi8>, vector<32xi8>
      return %0, %1 : vector<32xi8>, vector<32xi8>
   }
}

// -----

aie.device(npu1) {
   // CHECK-LABEL: func @unaligned_read
   // CHECK: %[[C24i32:.*]] = arith.constant 24 : i32
   // CHECK: %[[C12i32:.*]] = arith.constant 12 : i32
   // CHECK: %[[C0:.*]] = arith.constant 0 : index
   // CHECK: %[[LV:.*]] = aievec.upd %{{.*}}[%[[C0]]] {index = 0 : i8, offset = 0 : i32} : memref<64xi32>, vector<16xi32>
   // CHECK: %[[LV0:.*]] = aievec.ext %[[LV]] {index = 0 : i8} : vector<16xi32>, vector<8xi32>
   // CHECK: %[[LV1:.*]] = aievec.ext %[[LV]] {index = 1 : i8} : vector<16xi32>, vector<8xi32>
   // CHECK: %[[R0:.*]] = aievec.shift %[[LV0]], %[[LV1]], %[[C12i32]] {isAcc = false} : vector<8xi32>, vector<8xi32>, i32, vector<8xi32>
   // CHECK: %[[R1:.*]] = aievec.shift %[[LV0]], %[[LV1]], %[[C24i32]] {isAcc = false} : vector<8xi32>, vector<8xi32>, i32, vector<8xi32>
   // CHECK: return %[[R0]], %[[R1]] : vector<8xi32>, vector<8xi32>
   func.func @unaligned_read(%m: memref<64xi32>) -> (vector<8xi32>, vector<8xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c3 = arith.constant 3 : index
      %c6 = arith.constant 6 : index
      %0 = vector.transfer_read %m[%c3], %c0_i32 : memref<64xi32>, vector<8xi32>
      %1 = vector.transfer_read %m[%c6], %c0_i32 : memref<64xi32>, vector<8xi32>
      return %0, %1 : vector<8xi32>, vector<8xi32>
   }
}

// -----

aie.device(npu1) {
   // CHECK-LABEL: func @unaligned_read
   // CHECK: %[[C12i32:.*]] = arith.constant 12 : i32
   // CHECK: %[[C6i32:.*]] = arith.constant 6 : i32
   // CHECK: %[[C0:.*]] = arith.constant 0 : index
   // CHECK: %[[LV:.*]] = aievec.upd %{{.*}}[%[[C0]]] {index = 0 : i8, offset = 0 : i32} : memref<64xi16>, vector<32xi16>
   // CHECK: %[[LV0:.*]] = aievec.ext %[[LV]] {index = 0 : i8} : vector<32xi16>, vector<16xi16>
   // CHECK: %[[LV1:.*]] = aievec.ext %[[LV]] {index = 1 : i8} : vector<32xi16>, vector<16xi16>
   // CHECK: %[[R0:.*]] = aievec.shift %[[LV0]], %[[LV1]], %[[C6i32]] {isAcc = false} : vector<16xi16>, vector<16xi16>, i32, vector<16xi16>
   // CHECK: %[[R1:.*]] = aievec.shift %[[LV0]], %[[LV1]], %[[C12i32]] {isAcc = false} : vector<16xi16>, vector<16xi16>, i32, vector<16xi16>
   // CHECK: return %[[R0]], %[[R1]] : vector<16xi16>, vector<16xi16>
   func.func @unaligned_read(%m: memref<64xi16>) -> (vector<16xi16>, vector<16xi16>) {
      %c0_i16 = arith.constant 0 : i16
      %c3 = arith.constant 3 : index
      %c6 = arith.constant 6 : index
      %0 = vector.transfer_read %m[%c3], %c0_i16 : memref<64xi16>, vector<16xi16>
      %1 = vector.transfer_read %m[%c6], %c0_i16 : memref<64xi16>, vector<16xi16>
      return %0, %1 : vector<16xi16>, vector<16xi16>
   }
}
