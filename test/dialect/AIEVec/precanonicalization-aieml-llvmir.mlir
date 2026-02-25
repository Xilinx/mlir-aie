// RUN: aie-opt %s -canonicalize-vector-for-aievec="aie-target=aie2 target-backend=llvmir" -split-input-file | FileCheck %s

// CHECK-LABEL: @scalar_extsi_to_broadcast_swap(
// CHECK-SAME: %[[SIN:.*]]: i8
func.func @scalar_extsi_to_broadcast_swap(%s: i8) -> vector<32xi32> {
    // CHECK: %[[BCAST:.*]] = vector.broadcast %[[SIN]] : i8 to vector<32xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[BCAST]] : vector<32xi8> to vector<32xi32>
    %0 = arith.extsi %s : i8 to i32
    %1 = vector.broadcast %0 : i32 to vector<32xi32>
    return %1 : vector<32xi32>
}

// -----

// CHECK-LABEL: @extsi_to_broadcast_swap(
// CHECK-SAME: %[[VIN:.*]]: vector<8xi8>
func.func @extsi_to_broadcast_swap(%v: vector<8xi8>) -> vector<4x8xi32> {
    // CHECK: %[[ZV:.*]] = ub.poison : vector<4x8xi8>
    // CHECK: %[[I0:.*]] = vector.insert %[[VIN]], %[[ZV]] [0] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[I1:.*]] = vector.insert %[[VIN]], %[[I0]] [1] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[I2:.*]] = vector.insert %[[VIN]], %[[I1]] [2] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[BC:.*]] = vector.insert %[[VIN]], %[[I2]] [3] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[BC]] : vector<4x8xi8> to vector<4x8xi32>
    %0 = arith.extsi %v : vector<8xi8> to vector<8xi32>
    %1 = vector.broadcast %0 : vector<8xi32> to vector<4x8xi32>
    return %1 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: @broadcast_to_insert(
// CHECK-SAME: %[[V:.*]]: vector<8xbf16>
func.func @broadcast_to_insert(%v: vector<8xbf16>) -> vector<1x4x8xbf16> {
    // CHECK: %[[ZV:.*]] = ub.poison : vector<4x8xbf16>
    // CHECK: %[[I0:.*]] = vector.insert %[[V]], %[[ZV]] [0] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[I1:.*]] = vector.insert %[[V]], %[[I0]] [1] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[I2:.*]] = vector.insert %[[V]], %[[I1]] [2] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[I3:.*]] = vector.insert %[[V]], %[[I2]] [3] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[BC:.*]] = vector.shape_cast %[[I3]] : vector<4x8xbf16> to vector<1x4x8xbf16>
    // CHECK: return %[[BC]] : vector<1x4x8xbf16>
    %0 = vector.broadcast %v : vector<8xbf16> to vector<1x4x8xbf16>
    return %0 : vector<1x4x8xbf16>
}
