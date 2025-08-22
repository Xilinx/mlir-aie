// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s

// CHECK-LABEL: func @vector_extract_broadcast_to_aievec_512(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi32>
func.func @vector_extract_broadcast_to_aievec_512(%a : vector<16xi32>) -> (vector<16xi32>, vector<16xi32>) {
  // CHECK: aievec.broadcast %[[A]] {idx = 0 : i8} : vector<16xi32>, vector<16xi32>
  %0 = vector.extract %a[0] : i32 from vector<16xi32>
  %1 = vector.broadcast %0 : i32 to vector<16xi32>
  // CHECK: aievec.broadcast %[[A]] {idx = 2 : i8} : vector<16xi32>, vector<16xi32>
  %2 = vector.extract %a[2] : i32 from vector<16xi32>
  %3 = vector.broadcast %2 : i32 to vector<16xi32>
  return %1, %3 : vector<16xi32>, vector<16xi32>
}

// CHECK-LABEL: func @vector_extract_broadcast_to_aievec_256(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @vector_extract_broadcast_to_aievec_256(%a : vector<16xbf16>) -> (vector<16xbf16>, vector<16xbf16>) {
  // CHECK: %[[CC1:.*]] = aievec.concat %[[A]], %[[A]] : vector<16xbf16>, vector<32xbf16>
  // CHECK: %[[BCAST1:.*]] = aievec.broadcast %[[CC1]] {idx = 0 : i8} : vector<32xbf16>, vector<32xbf16>
  // CHECK: %[[EXT1:.*]] = aievec.ext %[[BCAST1]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  %0 = vector.extract %a[0] : bf16 from vector<16xbf16>
  %1 = vector.broadcast %0 : bf16 to vector<16xbf16>
  // CHECK: %[[BCAST2:.*]] = aievec.broadcast %[[CC1]] {idx = 2 : i8} : vector<32xbf16>, vector<32xbf16>
  // CHECK: %[[EXT2:.*]] = aievec.ext %[[BCAST2]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  %2 = vector.extract %a[2] : bf16 from vector<16xbf16>
  %3 = vector.broadcast %2 : bf16 to vector<16xbf16>
  return %1, %3 : vector<16xbf16>, vector<16xbf16>
}

// CHECK-LABEL: func @vector_extract_broadcast_to_aievec_1024(
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xi32>
func.func @vector_extract_broadcast_to_aievec_1024(%a : vector<32xi32>) -> (vector<32xi32>, vector<32xi32>) {
  // CHECK: %[[EXT1:.*]] = aievec.ext %[[A]] {index = 0 : i8} : vector<32xi32>, vector<16xi32>
  // CHECK: %[[BCAST1:.*]] = aievec.broadcast %[[EXT1]] {idx = 0 : i8} : vector<16xi32>, vector<16xi32>
  // CHECK: %[[CC1:.*]] = aievec.concat %[[BCAST1]], %[[BCAST1]] : vector<16xi32>, vector<32xi32>
  %0 = vector.extract %a[0] : i32 from vector<32xi32>
  %1 = vector.broadcast %0 : i32 to vector<32xi32>
  // CHECK: %[[BCAST2:.*]] = aievec.broadcast %[[EXT1]] {idx = 2 : i8} : vector<16xi32>, vector<16xi32>
  // CHECK: %[[CC2:.*]] = aievec.concat %[[BCAST2]], %[[BCAST2]] : vector<16xi32>, vector<32xi32>
  %2 = vector.extract %a[2] : i32 from vector<32xi32>
  %3 = vector.broadcast %2 : i32 to vector<32xi32>
  return %1, %3 : vector<32xi32>, vector<32xi32>
}

// CHECK-LABEL: func @vector_broadcast_from_scalar(
func.func @vector_broadcast_from_scalar(%a : i32, %b :bf16) -> (vector<16xi32>, vector<32xi32>, vector<16xbf16>, vector<32xbf16>) {
  // CHECK: %[[BCAST1:.*]] = aievec.broadcast_scalar %arg0 : i32, vector<16xi32>
  %0 = vector.broadcast %a : i32 to vector<16xi32>
  // CHECK: %[[CC:.*]] = aievec.concat %[[BCAST1]], %[[BCAST1]] : vector<16xi32>, vector<32xi32>
  %1 = vector.broadcast %a : i32 to vector<32xi32>
  // CHECK: %[[BCAST2:.*]] = aievec.broadcast_scalar %arg1 : bf16, vector<32xbf16>
  %3 = vector.broadcast %b : bf16 to vector<32xbf16>
  // CHECK: %[[EXT1:.*]] = aievec.ext %[[BCAST2]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  %2 = vector.broadcast %b : bf16 to vector<16xbf16>
  return %0, %1, %2, %3 : vector<16xi32>, vector<32xi32>, vector<16xbf16>, vector<32xbf16>
}

