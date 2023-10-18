// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm | FileCheck %s

func.func @matmul(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>,
                  %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16>
                                  into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: llvm.func @llvm.aie2.bf.mac16.conf(vector<32xbf16>,
// CHECK-SAME:  vector<32xbf16>, vector<16xf32>) -> vector<16xf32>
// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>
// CHECK-SAME: %[[B:.*]]: vector<8x4xbf16>
// CHECK-SAME: %[[C:.*]]: vector<4x4xf32>
// CHECK:      %[[FA:.*]] = builtin.unrealized_conversion_cast %[[A]] :
// CHECK-SAME:                      vector<4x8xbf16> to vector<32xbf16>
// CHECK:      %[[FB:.*]] = builtin.unrealized_conversion_cast %[[B]] :
// CHECK-SAME:                      vector<8x4xbf16> to vector<32xbf16>
// CHECK:      %[[FC:.*]] = builtin.unrealized_conversion_cast %[[C]] :
// CHECK-SAME:                      vector<4x4xf32> to vector<16xf32>
// CHECK:      %[[FR:.*]] = llvm.call @llvm.aie2.bf.mac16.conf(
// CHECK-SAME:                %[[FA]], %[[FB]], %[[FC]]) :
// CHECK-SAME:                (vector<32xbf16>, vector<32xbf16>, vector<16xf32>)
// CHECK-SAME:                -> vector<16xf32>
// CHECK:      %[[R:.*]] = builtin.unrealized_conversion_cast %[[FR]] :
// CHECK-SAME:                      vector<16xf32> to vector<4x4xf32>
// CHECK:      return %[[R]] : vector<4x4xf32>
