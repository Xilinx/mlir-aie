// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s

// CHECK-LABEL: func @vector_extract_broadcast_to_aievec(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi32>
func.func @vector_extract_broadcast_to_aievec(%a : vector<16xi32>) -> (vector<16xi32>, vector<16xi32>) {
  // CHECK: aievec.broadcast %[[A]] {idx = 0 : i8} : vector<16xi32>, vector<16xi32>
  %0 = vector.extract %a[0] : vector<16xi32>
  %1 = vector.broadcast %0 : i32 to vector<16xi32>
  // CHECK: aievec.broadcast %[[A]] {idx = 2 : i8} : vector<16xi32>, vector<16xi32>
  %2 = vector.extract %a[2] : vector<16xi32>
  %3 = vector.broadcast %2 : i32 to vector<16xi32>
  return %1, %3 : vector<16xi32>, vector<16xi32>
}
