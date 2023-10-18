// RUN: aie-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @matmul_i8i4
// CHECK-SAME: %[[A:.*]]: vector<4x16xi8>
// CHECK-SAME: %[[B:.*]]: vector<16x8xi4>
// CHECK-SAME: %[[C:.*]]: vector<4x8xi32>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<4x16xi8>, vector<16x8xi4> into vector<4x8xi32>
// CHECK: return %[[RES]] : vector<4x8xi32>
func.func @matmul_i8i4(%A : vector<4x16xi8>, %B : vector<16x8xi4>,
                       %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x16xi8>, vector<16x8xi4>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: @matmul_i8i8
// CHECK-SAME: %[[A:.*]]: vector<4x8xi8>
// CHECK-SAME: %[[B:.*]]: vector<8x8xi8>
// CHECK-SAME: %[[C:.*]]: vector<4x8xi32>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
// CHECK: return %[[RES]] : vector<4x8xi32>
func.func @matmul_i8i8(%A : vector<4x8xi8>, %B : vector<8x8xi8>,
                       %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xi8>, vector<8x8xi8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: @matmul_i16i8a
// CHECK-SAME: %[[A:.*]]: vector<4x4xi16>
// CHECK-SAME: %[[B:.*]]: vector<4x8xi8>
// CHECK-SAME: %[[C:.*]]: vector<4x8xi32>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<4x4xi16>, vector<4x8xi8> into vector<4x8xi32>
// CHECK: return %[[RES]] : vector<4x8xi32>
func.func @matmul_i16i8a(%A : vector<4x4xi16>, %B : vector<4x8xi8>,
                         %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x4xi16>, vector<4x8xi8>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: @matmul_i16i16a
// CHECK-SAME: %[[A:.*]]: vector<4x2xi16>
// CHECK-SAME: %[[B:.*]]: vector<2x8xi16>
// CHECK-SAME: %[[C:.*]]: vector<4x8xi32>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<4x2xi16>, vector<2x8xi16> into vector<4x8xi32>
// CHECK: return %[[RES]] : vector<4x8xi32>
func.func @matmul_i16i16a(%A : vector<4x2xi16>, %B : vector<2x8xi16>,
                          %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x2xi16>, vector<2x8xi16>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: @matmul_i16i8b
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>
// CHECK-SAME: %[[B:.*]]: vector<8x8xi8>
// CHECK-SAME: %[[C:.*]]: vector<2x8xi64>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<2x8xi16>, vector<8x8xi8> into vector<2x8xi64>
// CHECK: return %[[RES]] : vector<2x8xi64>
func.func @matmul_i16i8b(%A : vector<2x8xi16>, %B : vector<8x8xi8>,
                         %C : vector<2x8xi64>) -> vector<2x8xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<2x8xi16>, vector<8x8xi8>
                                  into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: @matmul_i16i8c
// CHECK-SAME: %[[A:.*]]: vector<4x8xi16>
// CHECK-SAME: %[[B:.*]]: vector<8x4xi8>
// CHECK-SAME: %[[C:.*]]: vector<4x4xi64>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<4x8xi16>, vector<8x4xi8> into vector<4x4xi64>
// CHECK: return %[[RES]] : vector<4x4xi64>
func.func @matmul_i16i8c(%A : vector<4x8xi16>, %B : vector<8x4xi8>,
                         %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xi16>, vector<8x4xi8>
                                  into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// CHECK-LABEL: @matmul_i16i16b
// CHECK-SAME: %[[A:.*]]: vector<2x4xi16>
// CHECK-SAME: %[[B:.*]]: vector<4x8xi16>
// CHECK-SAME: %[[C:.*]]: vector<2x8xi64>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<2x4xi16>, vector<4x8xi16> into vector<2x8xi64>
// CHECK: return %[[RES]] : vector<2x8xi64>
func.func @matmul_i16i16b(%A : vector<2x4xi16>, %B : vector<4x8xi16>,
                          %C : vector<2x8xi64>) -> vector<2x8xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<2x4xi16>, vector<4x8xi16>
                                  into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: @matmul_i16i16c
// CHECK-SAME: %[[A:[a-zA-Z0-9]+]]: vector<4x4xi16>
// CHECK-SAME: %[[B:[a-zA-Z0-9]+]]: vector<4x4xi16>
// CHECK-SAME: %[[C:[a-zA-Z0-9]+]]: vector<4x4xi64>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<4x4xi16>, vector<4x4xi16> into vector<4x4xi64>
// CHECK: return %[[RES]] : vector<4x4xi64>
func.func @matmul_i16i16c(%A : vector<4x4xi16>, %B : vector<4x4xi16>,
                          %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<4x4xi16>, vector<4x4xi16>
                                  into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// CHECK-LABEL: @matmul_i32i16
// CHECK-SAME: %[[A:.*]]: vector<4x2xi32>
// CHECK-SAME: %[[B:.*]]: vector<2x4xi16>
// CHECK-SAME: %[[C:.*]]: vector<4x4xi64>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<4x2xi32>, vector<2x4xi16> into vector<4x4xi64>
// CHECK: return %[[RES]] : vector<4x4xi64>
func.func @matmul_i32i16(%A : vector<4x2xi32>, %B : vector<2x4xi16>,
                         %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<4x2xi32>, vector<2x4xi16>
                                  into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// CHECK-LABEL: @matmul_bf16
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>
// CHECK-SAME: %[[B:.*]]: vector<8x4xbf16>
// CHECK-SAME: %[[C:.*]]: vector<4x4xf32>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[RES]] : vector<4x4xf32>
func.func @matmul_bf16(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>,
                       %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16>
                                  into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}
