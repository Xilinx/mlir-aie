// RUN: aie-opt %s -canonicalize-vector-for-aievec="aie-target=aie2 target-backend=llvmir" -split-input-file | FileCheck %s

// CHECK-LABEL: @extsi_to_broadcast_swap(
// CHECK-SAME: %[[VIN:.*]]: vector<8xi8>
func.func @extsi_to_broadcast_swap(%v: vector<8xi8>) -> vector<4x8xi32> {
    // CHECK: %[[BCAST:.*]] = vector.broadcast %[[VIN]] : vector<8xi8> to vector<4x8xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[BCAST]] : vector<4x8xi8> to vector<4x8xi32>
    %0 = arith.extsi %v : vector<8xi8> to vector<8xi32>
    %1 = vector.broadcast %0 : vector<8xi32> to vector<4x8xi32>
    return %1 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: @scalar_extsi_to_broadcast_swap(
// CHECK-SAME: %[[SIN:.*]]: i8
func.func @scalar_extsi_to_broadcast_swap(%s: i8) -> vector<32xi32> {
    // CHECK: %[[BCAST:.*]] = vector.broadcast %[[SIN]] : i8 to vector<32xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[BCAST]] : vector<32xi8> to vector<32xi32>
    %0 = arith.extsi %s : i8 to i32
    %1 = vector.broadcast %0 : i32 to vector<32xi32>
    return %1 : vector<32xi32>
}