// RUN: aie-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @single_contraction(%A : tensor<16x24xf32>, %B : tensor<24x16xf32>,
                              %C : tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
                       iterator_types = ["parallel", "parallel", "reduction"]}
                      ins(%A, %B : tensor<16x24xf32>, tensor<24x16xf32>)
                      outs(%C : tensor<16x16xf32>) {
          ^bb0(%in_a : f32, %in_b : f32, %out_c : f32):
            %0 = arith.mulf %in_a, %in_b : f32
            %1 = arith.addf %out_c, %0 : f32
            linalg.yield %1 : f32
        } -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0 : !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    %1 = transform.structured.vectorize_contraction %0
}

// CHECK-DAG: #[[NULLMAP:.*]] = affine_map<() -> ()>
// CHECK-DAG: #[[AMAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[BMAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[CMAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: @single_contraction
// CHECK-SAME: %[[A:.*]]: tensor<16x24xf32>
// CHECK-SAME: %[[B:.*]]: tensor<24x16xf32>
// CHECK-SAME: %[[C:.*]]: tensor<16x16xf32>
// CHECK: %[[AM:.*]] = bufferization.to_memref %[[A]] : memref<16x24xf32>
// CHECK: %[[AVM:.*]] = vector.type_cast %[[AM]] : memref<16x24xf32> to memref<vector<16x24xf32>>
// CHECK: %[[AV:.*]] = bufferization.to_tensor %[[AVM]] : memref<vector<16x24xf32>>
// CHECK: %[[BM:.*]] = bufferization.to_memref %[[B]] : memref<24x16xf32>
// CHECK: %[[BVM:.*]] = vector.type_cast %[[BM]] : memref<24x16xf32> to memref<vector<24x16xf32>>
// CHECK: %[[BV:.*]] = bufferization.to_tensor %[[BVM]] : memref<vector<24x16xf32>>
// CHECK: %[[CM:.*]] = bufferization.to_memref %[[C]] : memref<16x16xf32>
// CHECK: %[[CVM:.*]] = vector.type_cast %[[CM]] : memref<16x16xf32> to memref<vector<16x16xf32>>
// CHECK: %[[CV:.*]] = bufferization.to_tensor %[[CVM]] : memref<vector<16x16xf32>>
// CHECK: %[[RV:.*]] = linalg.generic {indexing_maps = [#[[NULLMAP]], #[[NULLMAP]], #[[NULLMAP]]],
// CHECK-SAME:                         iterator_types = []}
// CHECK-SAME:                        ins(%[[AV]], %[[BV]] : tensor<vector<16x24xf32>>, tensor<vector<24x16xf32>>)
// CHECK-SAME:                        outs(%[[CV]] : tensor<vector<16x16xf32>>) {
// CHECK:                 ^bb0(%[[IN0:.*]]: vector<16x24xf32>, %[[IN1:.*]]: vector<24x16xf32>, %[[OUT:.*]]: vector<16x16xf32>):
// CHECK:                   %[[RES:.*]] = vector.contract {indexing_maps = [#[[AMAP]], #[[BMAP]], #[[CMAP]]],
// CHECK-SAME:                                             iterator_types = ["parallel", "parallel", "reduction"],
// CHECK-SAME:                                             kind = #vector.kind<add>} %[[IN0]], %[[IN1]], %[[OUT]]
// CHECK-SAME:                                            : vector<16x24xf32>, vector<24x16xf32> into vector<16x16xf32>
// CHECK:                   linalg.yield %[[RES]] : vector<16x16xf32>
// CHECK:                  } -> tensor<vector<16x16xf32>>
// CHECK: %[[RVM:.*]] = bufferization.to_memref %[[RV]] : memref<vector<16x16xf32>>
// CHECK: %[[RM:.*]] = vector.type_cast %[[RVM]] : memref<vector<16x16xf32>> to memref<16x16xf32>
// CHECK: %[[R:.*]] = bufferization.to_tensor %[[RM]] : memref<16x16xf32>
// CHECK: return %[[R]] : tensor<16x16xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

func.func @multiple_parallel_contraction(%A : tensor<8x8x16x24xf32>, %B : tensor<8x8x24x16xf32>,
                                         %C : tensor<8x8x16x16xf32>) -> tensor<8x8x16x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
                      ins(%A, %B : tensor<8x8x16x24xf32>, tensor<8x8x24x16xf32>)
                      outs(%C : tensor<8x8x16x16xf32>) {
          ^bb0(%in_a : f32, %in_b : f32, %out_c : f32):
            %0 = arith.mulf %in_a, %in_b : f32
            %1 = arith.addf %out_c, %0 : f32
            linalg.yield %1 : f32
        } -> tensor<8x8x16x16xf32>
  return %0 : tensor<8x8x16x16xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0 : !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    %1 = transform.structured.vectorize_contraction %0
}

// CHECK-DAG: #[[PMAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[AMAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[BMAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[CMAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: @multiple_parallel_contraction
// CHECK-SAME: %[[A:.*]]: tensor<8x8x16x24xf32>
// CHECK-SAME: %[[B:.*]]: tensor<8x8x24x16xf32>
// CHECK-SAME: %[[C:.*]]: tensor<8x8x16x16xf32>
// CHECK: %[[AM:.*]] = bufferization.to_memref %[[A]] : memref<8x8x16x24xf32>
// CHECK: %[[AVM:.*]] = vector.type_cast %[[AM]] : memref<8x8x16x24xf32> to memref<8x8xvector<16x24xf32>>
// CHECK: %[[AV:.*]] = bufferization.to_tensor %[[AVM]] : memref<8x8xvector<16x24xf32>>
// CHECK: %[[BM:.*]] = bufferization.to_memref %[[B]] : memref<8x8x24x16xf32>
// CHECK: %[[BVM:.*]] = vector.type_cast %[[BM]] : memref<8x8x24x16xf32> to memref<8x8xvector<24x16xf32>>
// CHECK: %[[BV:.*]] = bufferization.to_tensor %[[BVM]] : memref<8x8xvector<24x16xf32>>
// CHECK: %[[CM:.*]] = bufferization.to_memref %[[C]] : memref<8x8x16x16xf32>
// CHECK: %[[CVM:.*]] = vector.type_cast %[[CM]] : memref<8x8x16x16xf32> to memref<8x8xvector<16x16xf32>>
// CHECK: %[[CV:.*]] = bufferization.to_tensor %[[CVM]] : memref<8x8xvector<16x16xf32>>
// CHECK: %[[RV:.*]] = linalg.generic {indexing_maps = [#[[PMAP]], #[[PMAP]], #[[PMAP]]],
// CHECK-SAME:                         iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[AV]], %[[BV]] : tensor<8x8xvector<16x24xf32>>, tensor<8x8xvector<24x16xf32>>)
// CHECK-SAME:                        outs(%[[CV]] : tensor<8x8xvector<16x16xf32>>) {
// CHECK:                 ^bb0(%[[IN0:.*]]: vector<16x24xf32>, %[[IN1:.*]]: vector<24x16xf32>, %[[OUT:.*]]: vector<16x16xf32>):
// CHECK:                   %[[RES:.*]] = vector.contract {indexing_maps = [#[[AMAP]], #[[BMAP]], #[[CMAP]]],
// CHECK-SAME:                                             iterator_types = ["parallel", "parallel", "reduction"],
// CHECK-SAME:                                             kind = #vector.kind<add>} %[[IN0]], %[[IN1]], %[[OUT]]
// CHECK-SAME:                                            : vector<16x24xf32>, vector<24x16xf32> into vector<16x16xf32>
// CHECK:                   linalg.yield %[[RES]] : vector<16x16xf32>
// CHECK:                  } -> tensor<8x8xvector<16x16xf32>>
// CHECK: %[[RVM:.*]] = bufferization.to_memref %[[RV]] : memref<8x8xvector<16x16xf32>>
// CHECK: %[[RM:.*]] = vector.type_cast %[[RVM]] : memref<8x8xvector<16x16xf32>> to memref<8x8x16x16xf32>
// CHECK: %[[R:.*]] = bufferization.to_tensor %[[RM]] : memref<8x8x16x16xf32>
// CHECK: return %[[R]] : tensor<8x8x16x16xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @packed_gemm(%A : tensor<16x8x4x8xbf16>, %B : tensor<8x16x8x4xbf16>,
                       %C : tensor<16x16x4x4xf32>) -> tensor<16x16x4x4xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]}
                      ins(%A, %B : tensor<16x8x4x8xbf16>, tensor<8x16x8x4xbf16>)
                      outs(%C : tensor<16x16x4x4xf32>) {
          ^bb0(%in_a : bf16, %in_b : bf16, %out_c : f32):
            %0 = arith.extf %in_a : bf16 to f32
            %1 = arith.extf %in_b : bf16 to f32
            %2 = arith.mulf %0, %1 : f32
            %3 = arith.addf %out_c, %2 : f32
            linalg.yield %3 : f32
        } -> tensor<16x16x4x4xf32>
  return %0 : tensor<16x16x4x4xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0 : !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    %1 = transform.structured.vectorize_contraction %0
}

// CHECK-DAG: #[[AMAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[BMAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[CMAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: @packed_gemm
// CHECK-SAME: %[[A:.*]]: tensor<16x8x4x8xbf16>
// CHECK-SAME: %[[B:.*]]: tensor<8x16x8x4xbf16>
// CHECK-SAME: %[[C:.*]]: tensor<16x16x4x4xf32>
// CHECK: %[[AM:.*]] = bufferization.to_memref %[[A]] : memref<16x8x4x8xbf16>
// CHECK: %[[AVM:.*]] = vector.type_cast %[[AM]] : memref<16x8x4x8xbf16> to memref<16x8xvector<4x8xbf16>>
// CHECK: %[[AV:.*]] = bufferization.to_tensor %[[AVM]] : memref<16x8xvector<4x8xbf16>>
// CHECK: %[[BM:.*]] = bufferization.to_memref %[[B]] : memref<8x16x8x4xbf16>
// CHECK: %[[BVM:.*]] = vector.type_cast %[[BM]] : memref<8x16x8x4xbf16> to memref<8x16xvector<8x4xbf16>>
// CHECK: %[[BV:.*]] = bufferization.to_tensor %[[BVM]] : memref<8x16xvector<8x4xbf16>>
// CHECK: %[[CM:.*]] = bufferization.to_memref %[[C]] : memref<16x16x4x4xf32>
// CHECK: %[[CVM:.*]] = vector.type_cast %[[CM]] : memref<16x16x4x4xf32> to memref<16x16xvector<4x4xf32>>
// CHECK: %[[CV:.*]] = bufferization.to_tensor %[[CVM]] : memref<16x16xvector<4x4xf32>>
// CHECK: %[[RV:.*]] = linalg.generic {indexing_maps = [#[[AMAP]], #[[BMAP]], #[[CMAP]]],
// CHECK-SAME:                         iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME:                        ins(%[[AV]], %[[BV]] : tensor<16x8xvector<4x8xbf16>>, tensor<8x16xvector<8x4xbf16>>)
// CHECK-SAME:                        outs(%[[CV]] : tensor<16x16xvector<4x4xf32>>) {
// CHECK:                 ^bb0(%[[IN0:.*]]: vector<4x8xbf16>, %[[IN1:.*]]: vector<8x4xbf16>, %[[OUT:.*]]: vector<4x4xf32>):
// CHECK:                   %[[IN0F32:.*]] = arith.extf %[[IN0]] : vector<4x8xbf16> to vector<4x8xf32>
// CHECK:                   %[[IN1F32:.*]] = arith.extf %[[IN1]] : vector<8x4xbf16> to vector<8x4xf32>
// CHECK:                   %[[RES:.*]] = vector.contract {indexing_maps = [#[[AMAP]], #[[BMAP]], #[[CMAP]]],
// CHECK-SAME:                                             iterator_types = ["parallel", "parallel", "reduction"],
// CHECK-SAME:                                             kind = #vector.kind<add>} %[[IN0F32]], %[[IN1F32]], %[[OUT]]
// CHECK-SAME:                                            : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
// CHECK:                   linalg.yield %[[RES]] : vector<4x4xf32>
// CHECK:                  } -> tensor<16x16xvector<4x4xf32>>
// CHECK: %[[RVM:.*]] = bufferization.to_memref %[[RV]] : memref<16x16xvector<4x4xf32>>
// CHECK: %[[RM:.*]] = vector.type_cast %[[RVM]] : memref<16x16xvector<4x4xf32>> to memref<16x16x4x4xf32>
// CHECK: %[[R:.*]] = bufferization.to_tensor %[[RM]] : memref<16x16x4x4xf32>
// CHECK: return %[[R]] : tensor<16x16x4x4xf32>
