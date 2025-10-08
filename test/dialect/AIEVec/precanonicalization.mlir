// RUN: aie-opt %s -canonicalize-vector-for-aievec -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @broadcast(
// CHECK-SAME: %[[MEM:.*]]: memref<?xi32>,
// CHECK-SAME: %[[POS:.*]]: index
func.func @broadcast(%m : memref<?xi32>, %pos : index) -> vector<8xi32> {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    %c0_i32 = arith.constant 0 : i32
    %i = affine.apply affine_map<(d0) -> (d0 + 5)>(%pos)
    // CHECK: %[[V:.*]] = vector.transfer_read %[[MEM]][%[[POS]]], %[[C0]] : memref<?xi32>, vector<8xi32>
    // CHECK: %[[E:.*]] = vector.extract %[[V]][5] : i32 from vector<8xi32>
    // CHECK: %[[B:.*]] = vector.broadcast %[[E]] : i32 to vector<8xi32>
    %v = vector.transfer_read %m[%i], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0) -> (0)>} : memref<?xi32>, vector<8xi32>
    // CHECK: return %[[B]] : vector<8xi32>
    return %v : vector<8xi32>
}

// -----

// CHECK: #[[IDXMAP:.*]] = affine_map<()[s0] -> (s0 + 24)>
// CHECK-LABEL: func.func @far_broadcast(
// CHECK-SAME: %[[MEM:.*]]: memref<?xi32>,
// CHECK-SAME: %[[POS:.*]]: index
func.func @far_broadcast(%m : memref<?xi32>, %pos : index) -> vector<8xi32> {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: %[[IDX:.*]] = affine.apply #[[IDXMAP]]()[%[[POS]]]
    %i = affine.apply affine_map<(d0) -> (d0 + 29)>(%pos)
    // CHECK: %[[V:.*]] = vector.transfer_read %[[MEM]][%[[IDX]]], %[[C0]] : memref<?xi32>, vector<8xi32>
    // CHECK: %[[E:.*]] = vector.extract %[[V]][5] : i32 from vector<8xi32>
    // CHECK: %[[B:.*]] = vector.broadcast %[[E]] : i32 to vector<8xi32>
    %v = vector.transfer_read %m[%i], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0) -> (0)>} : memref<?xi32>, vector<8xi32>
    // CHECK: return %[[B]] : vector<8xi32>
    return %v : vector<8xi32>
}

// -----

// CHECK-LABEL: func.func @unaligned_transfer_read(
// CHECK-SAME: %[[MEM:.*]]: memref<1024xi32>,
// CHECK-SAME: %[[POS:.*]]: index
func.func @unaligned_transfer_read(%m : memref<1024xi32>, %pos : index) -> vector<8xi32> {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
    %c0_i32 = arith.constant 0 : i32
    %i = affine.apply affine_map<(d0) -> (d0 + 5)>(%pos)
    // CHECK-DAG: %[[LV:.*]] = vector.transfer_read %[[MEM]][%[[POS]]], %[[C0]] : memref<1024xi32>, vector<16xi32>
    // CHECK-DAG: %[[AV:.*]] = vector.extract_strided_slice %[[LV]] {offsets = [5], sizes = [8], strides = [1]} : vector<16xi32> to vector<8xi32>
    %v = vector.transfer_read %m[%i], %c0_i32 : memref<1024xi32>, vector<8xi32>
    // CHECK: return %[[AV]] : vector<8xi32>
    return %v : vector<8xi32>
}

// -----

// CHECK-LABEL: func.func @rank_zero_transfer_read(
// CHECK-SAME: %[[MEM:.*]]: memref<i16>
func.func @rank_zero_transfer_read(%m : memref<i16>) -> vector<16xi16> {
    %c0_i16 = arith.constant 0 : i16
    // CHECK-DAG: %[[C0idx:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C0i16:.*]] = arith.constant 0 : i16
    // CHECK-DAG: %[[EXPMEM:.*]] = memref.expand_shape %[[MEM]] [] output_shape [1] : memref<i16> into memref<1xi16>
    // CHECK: %[[LV:.*]] = vector.transfer_read %[[EXPMEM]][%[[C0idx]]], %[[C0i16]] : memref<1xi16>, vector<16xi16>
    // CHECK: %[[E:.*]] = vector.extract %[[LV]][0] : i16 from vector<16xi16>
    // CHECK: %[[B:.*]] = vector.broadcast %[[E]] : i16 to vector<16xi16>
    %v = vector.transfer_read %m[], %c0_i16 {in_bounds = [true], permutation_map = affine_map<()->(0)>} : memref<i16>, vector<16xi16>
    // CHECK: return %[[B]] : vector<16xi16>
    return %v : vector<16xi16>
}

// -----

// CHECK-LABEL: func.func @extsi_hoisting_through_extract_n_bcast(
// CHECK-SAME: %[[VEC:.*]]: vector<16xi8>
func.func @extsi_hoisting_through_extract_n_bcast(%v : vector<16xi8>)
                                                            -> vector<32xi32> {
// CHECK: %[[EXV:.*]] = arith.extsi %[[VEC]] : vector<16xi8> to vector<16xi32>
// CHECK: %[[EXS:.*]] = vector.extract %[[EXV]][7] : i32 from vector<16xi32>
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[EXS]] : i32 to vector<32xi32>
// CHECK: return %[[BCAST]] : vector<32xi32>
  %si8 = vector.extract %v[7] : i8 from vector<16xi8>
  %vi8 = vector.broadcast %si8 : i8 to vector<32xi8>
  %vi32 = arith.extsi %vi8 : vector<32xi8> to vector<32xi32>
  return %vi32 : vector<32xi32>
}

// -----

// CHECK-LABEL: func.func @extsi_hoisting_through_extract_strided_slice(
// CHECK-SAME: %[[MEM:.*]]: memref<?xi8>
func.func @extsi_hoisting_through_extract_strided_slice(%m : memref<?xi8>)
                                                            -> vector<16xi32> {
// CHECK-DAG: %[[C0i8:.*]] = arith.constant 0 : i8
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
// CHECK: %[[VEC:.*]] = vector.transfer_read %[[MEM]][%[[C0]]], %[[C0i8]] :
// CHECK-SAME:            memref<?xi8>, vector<32xi8>
// CHECK: %[[EXV:.*]] = arith.extsi %[[VEC]] : vector<32xi8> to vector<32xi32>
// CHECK: %[[SLC:.*]] = vector.extract_strided_slice %[[EXV]]
// CHECK-SAME:             {offsets = [3], sizes = [16], strides = [1]} :
// CHECK-SAME:             vector<32xi32> to vector<16xi32>
// CHECK: return %[[SLC]] : vector<16xi32>
    %v = vector.transfer_read %m[%c0], %c0_i8 : memref<?xi8>, vector<32xi8>
    %slice = vector.extract_strided_slice %v
                {offsets = [3], sizes = [16], strides = [1]} :
                vector<32xi8> to vector<16xi8>
    %vi32 = arith.extsi %slice : vector<16xi8> to vector<16xi32>
    return %vi32 : vector<16xi32>
}
