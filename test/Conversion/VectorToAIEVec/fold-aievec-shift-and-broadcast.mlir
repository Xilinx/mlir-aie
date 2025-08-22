// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s

func.func @fold_aievec_shift_and_broadcast(%a: memref<48xi8>) -> (vector<32xi8>, vector<32xi8>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : i32
    %c22 = arith.constant 22 : i32
    %0 = aievec.upd %a[%c0] {index = 0 : i8, offset = 0 : i32} : memref<48xi8>, vector<32xi8>
    %1 = aievec.shift %0, %0, %c16 {isAcc = false} : vector<32xi8>, vector<32xi8>, i32, vector<32xi8>
    %2 = aievec.broadcast %1 {idx = 0 : i8} : vector<32xi8>, vector<32xi8>
    %3 = aievec.shift %0, %0, %c22 {isAcc = false} : vector<32xi8>, vector<32xi8>, i32, vector<32xi8>
    %4 = aievec.broadcast %3 {idx = 2 : i8} : vector<32xi8>, vector<32xi8>
    return %2, %4 : vector<32xi8>, vector<32xi8>
}
// CHECK-LABEL: func @fold_aievec_shift_and_broadcast   
// CHECK      :    %[[C0:.*]] = arith.constant 0 : index
// CHECK      :    %[[T0:.*]] = aievec.upd {{.*}}[%[[C0:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<48xi8>, vector<32xi8>
// CHECK-NOT  :    aievec.shift %[[T0:.*]]
// CHECK      :    %[[T1:.*]] = aievec.broadcast %[[T0:.*]] {idx = 16 : i8} : vector<32xi8>, vector<32xi8>
// CHECK      :    %[[T2:.*]] = aievec.broadcast %[[T0:.*]] {idx = 24 : i8} : vector<32xi8>, vector<32xi8>
// CHECK      :    return %[[T1:.*]], %[[T2:.*]] : vector<32xi8>, vector<32xi8>

