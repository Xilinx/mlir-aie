// RUN: aie-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @packed_gemm(%A : tensor<16x8x4x8xbf16>, %B : tensor<8x16x8x4xbf16>,
                       %C : tensor<16x16x4x4xf32>, %D : tensor<16x16x4x4xf32>)
                            -> tensor<16x16x4x4xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map2],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]}
                      ins(%A, %B, %C : tensor<16x8x4x8xbf16>, tensor<8x16x8x4xbf16>, tensor<16x16x4x4xf32>)
                      outs(%D : tensor<16x16x4x4xf32>) {
          ^bb0(%in_a : bf16, %in_b : bf16, %in_c : f32, %out_c : f32):
            %0 = arith.extf %in_a : bf16 to f32
            %1 = arith.extf %in_b : bf16 to f32
            %2 = arith.mulf %0, %1 : f32
            %3 = arith.addf %out_c, %2 : f32
            %4 = arith.addf %3, %in_c : f32
            linalg.yield %3 : f32
        } -> tensor<16x16x4x4xf32>
  return %0 : tensor<16x16x4x4xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0 : !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    // expected-error @+1 {{payload is not a contraction.}}
    %1 = transform.structured.vectorize_contraction %0
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @packed_gemm(%A : tensor<16x8x4x8xbf16>, %B : tensor<8x16x8x4xbf16>,
                       %C : tensor<16x16x4x4xf32>) -> tensor<16x16x4x4xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "parallel"]}
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
    // expected-error @+1 {{linalg.generic op innermost iterators don't correspond with a gemm-like contraction.}}
    %1 = transform.structured.vectorize_contraction %0
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d3)>

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
    // expected-error @+1 {{linalg.generic op innermost indexing maps don't correspond with a gemm-like contraction.}}
    %1 = transform.structured.vectorize_contraction %0
}

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
            %4 = arith.addf %2, %3 : f32
            linalg.yield %4 : f32
        } -> tensor<16x16x4x4xf32>
  return %0 : tensor<16x16x4x4xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0 : !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    // expected-error @+1 {{linalg.generic op payload does not correspond with a vectorizable contraction.}}
    %1 = transform.structured.vectorize_contraction %0
}

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
            %2 = arith.addf %0, %1 : f32
            linalg.yield %2 : f32
        } -> tensor<16x16x4x4xf32>
  return %0 : tensor<16x16x4x4xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0 : !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    // expected-error @+1 {{linalg.generic op payload does not correspond with a vectorizable contraction.}}
    %1 = transform.structured.vectorize_contraction %0
}

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
            %2 = arith.subf %0, %1 : f32
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %out_c, %3 : f32
            linalg.yield %4 : f32
        } -> tensor<16x16x4x4xf32>
  return %0 : tensor<16x16x4x4xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0 : !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    // expected-error @+1 {{linalg.generic op payload does not correspond with a vectorizable contraction.}}
    %1 = transform.structured.vectorize_contraction %0
}
