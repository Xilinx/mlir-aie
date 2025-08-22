// RUN: aie-opt %s -canonicalize-vector-for-aievec=aie-target=aie2 -canonicalize -cse -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @multidim_vector_transfer(
// CHECK-SAME: %[[IMEM:[a-zA-Z0-9]+]]: memref<64x64x4x8xbf16>,
// CHECK-SAME: %[[OMEM:[a-zA-Z0-9]+]]: memref<64x64x4x8xbf16>) {
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[C0bf16:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:   affine.for %[[I:.*]] = 0 to 64 {
// CHECK:       affine.for %[[J:.*]] = 0 to 64 {
// CHECK:           %[[FIMEM:.*]] = memref.collapse_shape %[[IMEM]]
// CHECK-SAME:              {{\[}}[0], [1], [2, 3]] :
// CHECK-SAME:              memref<64x64x4x8xbf16> into memref<64x64x32xbf16>
// CHECK:           %[[FIV:.*]] = vector.transfer_read %[[FIMEM]]
// CHECK-SAME:              [%[[I]], %[[J]], %[[C0]]], %[[C0bf16]] {in_bounds = [true]} :
// CHECK-SAME:              memref<64x64x32xbf16>, vector<32xbf16>
// CHECK:           %[[IV:.*]] = vector.shape_cast %[[FIV]] :
// CHECK-SAME:                          vector<32xbf16> to vector<4x8xbf16>
// CHECK:           %[[IV2:.*]] = arith.addf %[[IV]], %[[IV]] : vector<4x8xbf16>
// CHECK:           %[[FOV:.*]] = vector.shape_cast %[[IV2]] :
// CHECK-SAME:                           vector<4x8xbf16> to vector<32xbf16>
// CHECK:           %[[FOMEM:.*]] = memref.collapse_shape %[[OMEM]]
// CHECK-SAME:              {{\[}}[0], [1], [2, 3]] :
// CHECK-SAME:              memref<64x64x4x8xbf16> into memref<64x64x32xbf16>
// CHECK:           vector.transfer_write %[[FOV]], %[[FOMEM]]
// CHECK-SAME:              [%[[I]], %[[J]], %[[C0]]] {in_bounds = [true]} :
// CHECK-SAME:              vector<32xbf16>, memref<64x64x32xbf16>
func.func @multidim_vector_transfer(%in : memref<64x64x4x8xbf16>,
                                    %out : memref<64x64x4x8xbf16>) {
  %c0 = arith.constant 0 : index
  %c0_bf16 = arith.constant 0.0 : bf16
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %v = vector.transfer_read %in[%i, %j, %c0, %c0], %c0_bf16 : memref<64x64x4x8xbf16>, vector<4x8xbf16>
      %v2 = arith.addf %v, %v : vector<4x8xbf16>
      vector.transfer_write %v2, %out[%i, %j, %c0, %c0] : vector<4x8xbf16>, memref<64x64x4x8xbf16>
    }
  }
  return
}

//
// -----
//

// CHECK: #[[IMIDXMAP:.*]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-LABEL: module {
// CHECK-LABEL: func.func @multidim_vector_transfer(
// CHECK-SAME: %[[IMEM:[a-zA-Z0-9]+]]: memref<64x64x32x8xbf16>,
// CHECK-SAME: %[[OMEM:[a-zA-Z0-9]+]]: memref<64x64x32x8xbf16>) {
// CHECK:   %[[C0bf16:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:   affine.for %[[I:.*]] = 0 to 64 {
// CHECK:     affine.for %[[J:.*]] = 0 to 64 {
// CHECK:       affine.for %[[K:.*]] = 0 to 32 step 4 {
// CHECK:           %[[IMIDX:.*]] = affine.apply #[[IMIDXMAP]](%[[K]])
// CHECK:           %[[FIMEM:.*]] = memref.collapse_shape %[[IMEM]]
// CHECK-SAME:              {{\[}}[0], [1], [2, 3]] :
// CHECK-SAME:              memref<64x64x32x8xbf16> into memref<64x64x256xbf16>
// CHECK:           %[[FIV:.*]] = vector.transfer_read %[[FIMEM]]
// CHECK-SAME:              [%[[I]], %[[J]], %[[IMIDX]]], %[[C0bf16]] :
// CHECK-SAME:              memref<64x64x256xbf16>, vector<32xbf16>
// CHECK:           %[[IV:.*]] = vector.shape_cast %[[FIV]] :
// CHECK-SAME:                          vector<32xbf16> to vector<4x8xbf16>
// CHECK:           %[[IV2:.*]] = arith.addf %[[IV]], %[[IV]] : vector<4x8xbf16>
// CHECK:           %[[FOV:.*]] = vector.shape_cast %[[IV2]] :
// CHECK-SAME:                           vector<4x8xbf16> to vector<32xbf16>
// CHECK:           %[[FOMEM:.*]] = memref.collapse_shape %[[OMEM]]
// CHECK-SAME:              {{\[}}[0], [1], [2, 3]] :
// CHECK-SAME:              memref<64x64x32x8xbf16> into memref<64x64x256xbf16>
// CHECK:           vector.transfer_write %[[FOV]], %[[FOMEM]]
// CHECK-SAME:              [%[[I]], %[[J]], %[[IMIDX]]] :
// CHECK-SAME:              vector<32xbf16>, memref<64x64x256xbf16>
func.func @multidim_vector_transfer(%in : memref<64x64x32x8xbf16>,
                                    %out : memref<64x64x32x8xbf16>) {
  %c0 = arith.constant 0 : index
  %c0_bf16 = arith.constant 0.0 : bf16
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 32 step 4 {
        %v = vector.transfer_read %in[%i, %j, %k, %c0], %c0_bf16 : memref<64x64x32x8xbf16>, vector<4x8xbf16>
        %v2 = arith.addf %v, %v : vector<4x8xbf16>
        vector.transfer_write %v2, %out[%i, %j, %k, %c0] : vector<4x8xbf16>, memref<64x64x32x8xbf16>
    }
    }
  }
  return
}

//
// -----
//

// CHECK: #[[IDXMAPA:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
// CHECK: #[[IDXMAPB:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
// CHECK: #[[IDXMAPC:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>

// CHECK-LABEL: func.func @vector_contract_permuted_b(
// CHECK-SAME: %[[VA:[a-zA-Z0-9]+]]: vector<1x1x4x8xbf16>,
// CHECK-SAME: %[[VB:[a-zA-Z0-9]+]]: vector<1x1x4x8xbf16>,
// CHECK-SAME: %[[VC:[a-zA-Z0-9]+]]: vector<1x1x4x4xf32>
func.func @vector_contract_permuted_b(%A : vector<1x1x4x8xbf16>,
                                      %B : vector<1x1x4x8xbf16>,
                                      %C : vector<1x1x4x4xf32>)
                                    -> vector<1x1x4x4xf32> {
  // CHECK: %[[TRB:.*]] = vector.transpose %[[VB]], [0, 1, 3, 2] :
  // CHECK-SAME:                 vector<1x1x4x8xbf16> to vector<1x1x8x4xbf16>
  // CHECK: %[[RES:.*]] = vector.contract {
  // CHECK-SAME:    indexing_maps = [#[[IDXMAPA]], #[[IDXMAPB]], #[[IDXMAPC]]],
  // CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction",
  // CHECK-SAME:                      "parallel", "parallel", "reduction"],
  // CHECK-SAME:    kind = #vector.kind<add>}
  // CHECK-SAME:    %[[VA]], %[[TRB]], %[[VC]] :
  // CHECK-SAME:          vector<1x1x4x8xbf16>, vector<1x1x8x4xbf16>
  // CHECK-SAME:          into vector<1x1x4x4xf32>
  %res = vector.contract {
              indexing_maps = [#map1, #map2, #map3],
              iterator_types = ["parallel", "parallel", "reduction",
                                "parallel", "parallel", "reduction"],
              kind = #vector.kind<add>} %A, %B, %C :
              vector<1x1x4x8xbf16>, vector<1x1x4x8xbf16> into vector<1x1x4x4xf32>
  return %res : vector<1x1x4x4xf32>
}

//
// -----
//

// CHECK: #[[IDXMAPA:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
// CHECK: #[[IDXMAPB:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
// CHECK: #[[IDXMAPC:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>

// CHECK-LABEL: func.func @vector_contract_permuted_b(
// CHECK-SAME: %[[VA:[a-zA-Z0-9]+]]: vector<1x1x4x8xbf16>,
// CHECK-SAME: %[[VB:[a-zA-Z0-9]+]]: vector<1x1x4x8xbf16>,
// CHECK-SAME: %[[VC:[a-zA-Z0-9]+]]: vector<1x1x4x4xf32>
func.func @vector_contract_permuted_b(%A : vector<1x1x4x8xbf16>,
                                      %B : vector<1x1x4x8xbf16>,
                                      %C : vector<1x1x4x4xf32>)
                                    -> vector<1x1x4x4xf32> {
  // CHECK: %[[LHS:.*]] = arith.extf %[[VA]] :
  // CHECK-SAME:                 vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
  // CHECK: %[[TRB:.*]] = vector.transpose %[[VB]], [0, 1, 3, 2] :
  // CHECK-SAME:                 vector<1x1x4x8xbf16> to vector<1x1x8x4xbf16>
  // CHECK: %[[RHS:.*]] = arith.extf %[[TRB]] :
  // CHECK-SAME:                 vector<1x1x8x4xbf16> to vector<1x1x8x4xf32>
  // CHECK: %[[RES:.*]] = vector.contract {
  // CHECK-SAME:    indexing_maps = [#[[IDXMAPA]], #[[IDXMAPB]], #[[IDXMAPC]]],
  // CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction",
  // CHECK-SAME:                      "parallel", "parallel", "reduction"],
  // CHECK-SAME:    kind = #vector.kind<add>}
  // CHECK-SAME:    %[[LHS]], %[[RHS]], %[[VC]] :
  // CHECK-SAME:          vector<1x1x4x8xf32>, vector<1x1x8x4xf32>
  // CHECK-SAME:          into vector<1x1x4x4xf32>
  %lhs = arith.extf %A : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
  %rhs = arith.extf %B : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
  %res = vector.contract {
              indexing_maps = [#map1, #map2, #map3],
              iterator_types = ["parallel", "parallel", "reduction",
                                "parallel", "parallel", "reduction"],
              kind = #vector.kind<add>} %lhs, %rhs, %C :
              vector<1x1x4x8xf32>, vector<1x1x4x8xf32> into vector<1x1x4x4xf32>
  return %res : vector<1x1x4x4xf32>
}