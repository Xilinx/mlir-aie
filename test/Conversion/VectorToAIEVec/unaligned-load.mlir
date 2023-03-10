// RUN: aie-opt %s --convert-vector-to-aievec | FileCheck %s
func.func @unaligned_read(%a: memref<48xi8>) -> (vector<32xi8>, vector<32xi8>) {
   %c0_i8 = arith.constant 0 : i8
   %c16 = arith.constant 16 : index
   %c34 = arith.constant 34 : index
   %0 = vector.transfer_read %a[%c16], %c0_i8 : memref<48xi8>, vector<32xi8>
   %1 = vector.transfer_read %a[%c34], %c0_i8 : memref<48xi8>, vector<32xi8>
   return %0, %1 : vector<32xi8>, vector<32xi8>
}

// CHECK-LABEL: func @unaligned_read
// CHECK      :    %[[C64:.*]] = arith.constant 64 : index
// CHECK      :    %[[C32:.*]] = arith.constant 32 : index
// CHECK      :    %[[C0:.*]] = arith.constant 0 : index
// CHECK      :    %[[T0:.*]] = aievec.upd {{.*}}[%[[C0:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<48xi8>, vector<32xi8>
// CHECK      :    %[[T1:.*]] = aievec.upd {{.*}}[%[[C32:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<48xi8>, vector<32xi8>
// CHECK      :    %[[T2:.*]] = aievec.shift %[[T0:.*]], %[[T1:.*]] {shift = 16 : i32} : vector<32xi8>, vector<32xi8>
// CHECK      :    %[[T3:.*]] = aievec.upd {{.*}}[%[[C64:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<48xi8>, vector<32xi8>
// CHECK      :    %[[T4:.*]] = aievec.shift %[[T1:.*]], %[[T3:.*]] {shift = 2 : i32} : vector<32xi8>, vector<32xi8>
// CHECK      :    return %[[T2:.*]], %[[T4:.*]] : vector<32xi8>, vector<32xi8>
