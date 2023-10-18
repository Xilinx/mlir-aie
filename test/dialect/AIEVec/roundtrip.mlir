// RUN: aie-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>
// CHECK-SAME: %[[B:.*]]: vector<8x4xbf16>
// CHECK-SAME: %[[C:.*]]: vector<4x4xf32>
// CHECK:      %[[RES:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] :
// CHECK-SAME: vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[RES]] : vector<4x4xf32>
func.func @matmul(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>,
                  %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16>
                                  into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}
