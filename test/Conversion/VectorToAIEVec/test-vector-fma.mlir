// RUN: aie-opt %s -convert-vector-to-aievec="aie-target=aie2" -split-input-file | FileCheck %s

// CHECK-LABEL: test_fma_bf16
// CHECK-SAME: %[[V0:[a-zA-Z0-9]+]]: vector<16xbf16>,
// CHECK-SAME: %[[V1:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[V2:.*]]: vector<16xbf16>)
func.func @test_fma_bf16(%v0: vector<16xbf16>,
                         %v1: vector<16xbf16>,
                         %v2: vector<16xbf16>) -> vector<16xbf16> {
  // CHECK: %[[ACC:.*]] = aievec.ups %[[V2]] {shift = 0 : i8}
  // CHECK-SAME:              : vector<16xbf16>, vector<16xf32>
  // CHECK: %[[FMA:.*]] = aievec.mac_elem %[[V0]], %[[V1]], %[[ACC]]
  // CHECK-SAME:              : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
  // CHECK: %[[RES:.*]] = aievec.srs %[[FMA]], %{{[a-zA-Z0-9]+}}
  // CHECK-SAME:              : vector<16xf32>, i32, vector<16xbf16>
  // CHECK: return %[[RES]] : vector<16xbf16>
  %0 = vector.fma %v0, %v1, %v2 : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// -----

// CHECK-LABEL: test_fma_f32
// CHECK-SAME: %[[V0:[a-zA-Z0-9]+]]: vector<16xbf16>,
// CHECK-SAME: %[[V1:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[V2:.*]]: vector<16xf32>)
func.func @test_fma_f32(%v0: vector<16xbf16>,
                        %v1: vector<16xbf16>,
                        %v2: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[RES:.*]] = aievec.mac_elem %[[V0]], %[[V1]], %[[V2]]
  // CHECK-SAME:              : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
  // CHECK: return %[[RES]] : vector<16xf32>
  %v0f32 = arith.extf %v0 : vector<16xbf16> to vector<16xf32>
  %v1f32 = arith.extf %v1 : vector<16xbf16> to vector<16xf32>
  %0 = vector.fma %v0f32, %v1f32, %v2 : vector<16xf32>
  return %0 : vector<16xf32>
}
