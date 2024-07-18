// RUN: aie-opt %s -convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir" | FileCheck %s

#map  = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d4, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d3, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d3, d4, d6, d7)>
func.func @matmul(%A : memref<1x1x4x8x4x8xbf16, 2 : i32>,
                  %B : memref<1x1x8x4x8x4xbf16, 2 : i32>,
                  %C : memref<1x1x8x8x4x4xf32, 2 : i32>) {
  %c0 = arith.constant 0 : index
  %c0_bf16 = arith.constant 0.0 : bf16
  %c0_f32 = arith.constant 0.0 : f32
  affine.for %i = 0 to 8 step 1 {
    affine.for %j = 0 to 8 step 1 {
      affine.for %k = 0 to 4 step 1 {
        %0 = vector.transfer_read %A[%c0, %c0, %k, %i, %c0, %c0], %c0_bf16
                {in_bounds = [true, true, true, true, true, true]}
                : memref<1x1x4x8x4x8xbf16, 2 : i32>, vector<1x1x1x1x4x8xbf16>
        %1 = vector.transfer_read %B[%c0, %c0, %j, %k, %c0, %c0], %c0_bf16
                {in_bounds = [true, true, true, true, true, true]}
                : memref<1x1x8x4x8x4xbf16, 2 : i32>, vector<1x1x1x1x8x4xbf16>
        %2 = vector.transfer_read %C[%c0, %c0, %j, %i, %c0, %c0], %c0_f32
                {in_bounds = [true, true, true, true, true, true]}
                : memref<1x1x8x8x4x4xf32, 2 : i32>, vector<1x1x1x1x4x4xf32>
        %3 = arith.extf %0 : vector<1x1x1x1x4x8xbf16> to vector<1x1x1x1x4x8xf32>
        %4 = arith.extf %1 : vector<1x1x1x1x8x4xbf16> to vector<1x1x1x1x8x4xf32>
        %5 = vector.contract {
                          indexing_maps = [#map, #map1, #map2],
                          iterator_types = ["parallel", "parallel", "reduction",
                                            "parallel", "parallel", "reduction",
                                            "parallel", "parallel", "reduction"],
                                            kind = #vector.kind<add>} %3, %4, %2
                              : vector<1x1x1x1x4x8xf32>, vector<1x1x1x1x8x4xf32>
                                into vector<1x1x1x1x4x4xf32>
        vector.transfer_write %5, %C[%c0, %c0, %j, %i, %c0, %c0]
                {in_bounds = [true, true, true, true, true, true]}
                : vector<1x1x1x1x4x4xf32>, memref<1x1x8x8x4x4xf32, 2 : i32>
      }
    }
  }
  return
}

// CHECK: #[[CMAP:.*]] = affine_map<(d0, d1) -> (d0 * 128 + d1 * 16)>
// CHECK: #[[AMAP:.*]] = affine_map<(d0, d1) -> (d0 * 256 + d1 * 32)>
// CHECK: #[[BMAP:.*]] = affine_map<(d0, d1) -> (d0 * 128 + d1 * 32)>
// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:        %[[A:.*]]: memref<1x1x4x8x4x8xbf16, 2 : i32>,
// CHECK-SAME:        %[[B:.*]]: memref<1x1x8x4x8x4xbf16, 2 : i32>,
// CHECK-SAME:        %[[C:.*]]: memref<1x1x8x8x4x4xf32, 2 : i32>) {
// CHECK:         %[[C0BF16:.*]] = arith.constant 0.0{{.*}} : bf16
// CHECK:         %[[C0F32:.*]] = arith.constant 0.0{{.*}} : f32
// CHECK:         %[[CSA:.*]] = memref.collapse_shape %[[A]] {{\[\[}}0, 1, 2, 3, 4, 5]]
// CHECK:         %[[CSB:.*]] = memref.collapse_shape %[[B]] {{\[\[}}0, 1, 2, 3, 4, 5]]
// CHECK:         %[[CSC:.*]] = memref.collapse_shape %[[C]] {{\[\[}}0, 1, 2, 3, 4, 5]]
// CHECK:         affine.for %[[I:.*]] = 0 to 8 {
// CHECK:           affine.for %[[J:.*]] = 0 to 8 {
// CHECK:             %[[CIDX:.*]] = affine.apply #[[CMAP]](%[[J]], %[[I]])
// CHECK:             affine.for %[[K:.*]] = 0 to 4 {
// CHECK:               %[[AIDX:.*]] = affine.apply #[[AMAP]](%[[K]], %[[I]])
// CHECK:               %[[FVA:.*]] = vector.transfer_read %[[CSA]][%[[AIDX]]],
// CHECK-SAME:                          %[[C0BF16]] {in_bounds = [true]}
// CHECK-SAME:                          : memref<1024xbf16, 2 : i32>, vector<32xbf16>
// CHECK:               %[[BIDX:.*]] = affine.apply #[[BMAP]](%[[J]], %[[K]])
// CHECK:               %[[FVB:.*]] = vector.transfer_read %[[CSB]][%[[BIDX]]],
// CHECK-SAME:                          %[[C0BF16]] {in_bounds = [true]}
// CHECK-SAME:                          : memref<1024xbf16, 2 : i32>, vector<32xbf16>
// CHECK:               %[[FVC:.*]] = vector.transfer_read %[[CSC]][%[[CIDX]]],
// CHECK-SAME:                          %[[C0F32]] {in_bounds = [true]}
// CHECK-SAME:                          : memref<1024xf32, 2 : i32>, vector<16xf32>
// CHECK:               %[[VC:.*]] = vector.shape_cast %[[FVC]] : vector<16xf32> to vector<4x4xf32>
// CHECK:               %[[VA:.*]] = vector.shape_cast %[[FVA]] : vector<32xbf16> to vector<4x8xbf16>
// CHECK:               %[[VB:.*]] = vector.shape_cast %[[FVB]] : vector<32xbf16> to vector<8x4xbf16>
// CHECK:               %[[MM:.*]] = aievec.matmul %[[VA]], %[[VB]], %[[VC]] :
// CHECK:               %[[R:.*]] = vector.shape_cast %[[MM]] : vector<4x4xf32> to vector<16xf32>
// CHECK:               vector.transfer_write %[[R]], %[[CSC]][%[[CIDX]]]
