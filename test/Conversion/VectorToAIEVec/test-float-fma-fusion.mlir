// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_fma_fusion_bf16_v16
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xbf16>
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_fma_fusion_bf16_v16(%a: vector<16xbf16>,
                                     %b: vector<16xbf16>,
                                     %c: vector<16xbf16>) -> vector<16xbf16> {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[UPS:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
    // CHECK: %[[FMA:.*]] = aievec.mac_elem %[[A]], %[[B]], %[[UPS]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
    // CHECK: %[[RES:.*]] = aievec.srs %[[FMA]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
    %0 = arith.mulf %a, %b : vector<16xbf16>
    %1 = arith.addf %0, %c : vector<16xbf16>
    // CHECK: return %[[RES]] : vector<16xbf16>
    return %1 : vector<16xbf16>
}

// -----

// Test commuted operand order: acc + mul
// CHECK-LABEL: func @test_fma_fusion_bf16_v16_commuted
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xbf16>
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_fma_fusion_bf16_v16_commuted(%a: vector<16xbf16>,
                                              %b: vector<16xbf16>,
                                              %c: vector<16xbf16>) -> vector<16xbf16> {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[UPS:.*]] = aievec.ups %[[C]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
    // CHECK: %[[FMA:.*]] = aievec.mac_elem %[[A]], %[[B]], %[[UPS]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
    // CHECK: %[[RES:.*]] = aievec.srs %[[FMA]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
    %0 = arith.mulf %a, %b : vector<16xbf16>
    %1 = arith.addf %c, %0 : vector<16xbf16>
    // CHECK: return %[[RES]] : vector<16xbf16>
    return %1 : vector<16xbf16>
}

// -----

// Test 32-lane bf16 FMA fusion (split into two 16-lane FMAs)
// CHECK-LABEL: func @test_fma_fusion_bf16_v32
func.func @test_fma_fusion_bf16_v32(%a: vector<32xbf16>,
                                     %b: vector<32xbf16>,
                                     %c: vector<32xbf16>) -> vector<32xbf16> {
    // CHECK-DAG: aievec.ext %{{.*}} {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
    // CHECK-DAG: aievec.ext %{{.*}} {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
    // CHECK: aievec.ups %{{.*}} {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
    // CHECK: aievec.mac_elem %{{.*}}, %{{.*}}, %{{.*}} : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
    // CHECK: aievec.srs %{{.*}}, %{{.*}} : vector<16xf32>, i32, vector<16xbf16>
    // CHECK: aievec.ups %{{.*}} {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
    // CHECK: aievec.mac_elem %{{.*}}, %{{.*}}, %{{.*}} : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
    // CHECK: aievec.srs %{{.*}}, %{{.*}} : vector<16xf32>, i32, vector<16xbf16>
    // CHECK: aievec.concat %{{.*}}, %{{.*}} : vector<16xbf16>, vector<32xbf16>
    %0 = arith.mulf %a, %b : vector<32xbf16>
    %1 = arith.addf %0, %c : vector<32xbf16>
    return %1 : vector<32xbf16>
}
