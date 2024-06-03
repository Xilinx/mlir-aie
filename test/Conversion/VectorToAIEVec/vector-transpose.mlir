// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" -split-input-file | FileCheck %s --check-prefix=CHECK

// CHECK-LABEL: transpose_i8_types
// CHECK-SAME: %[[V0:.*]]: vector<4x16xi8>,
// CHECK-SAME: %[[V1:.*]]: vector<16x4xi8>,
// CHECK-SAME: %[[V2:.*]]: vector<8x8xi8>
func.func @transpose_i8_types(%v0 : vector<4x16xi8>,
                              %v1 : vector<16x4xi8>,
                              %v2 : vector<8x8xi8>)
    -> (vector<4x16xi8>, vector<16x4xi8>, vector<8x8xi8>) {
  // CHECK: %[[FV0:.*]] = vector.shape_cast %[[V0]] : vector<4x16xi8> to vector<64xi8>
  // CHECK: %[[FR0:.*]] = aievec.shuffle %[[FV0]] [t8_4x16] : vector<64xi8>
  // CHECK: %[[R0:.*]] = vector.shape_cast %[[FR0]] : vector<64xi8> to vector<16x4xi8>
  %v0t = vector.transpose %v0, [1, 0] : vector<4x16xi8> to vector<16x4xi8>
  // CHECK: %[[FV1:.*]] = vector.shape_cast %[[V1]] : vector<16x4xi8> to vector<64xi8>
  // CHECK: %[[FR1:.*]] = aievec.shuffle %[[FV1]] [t8_16x4] : vector<64xi8>
  // CHECK: %[[R1:.*]] = vector.shape_cast %[[FR1]] : vector<64xi8> to vector<4x16xi8>
  %v1t = vector.transpose %v1, [1, 0] : vector<16x4xi8> to vector<4x16xi8>
  // CHECK: %[[FV2:.*]] = vector.shape_cast %[[V2]] : vector<8x8xi8> to vector<64xi8>
  // CHECK: %[[FR2:.*]] = aievec.shuffle %[[FV2]] [t8_8x8] : vector<64xi8>
  // CHECK: %[[R2:.*]] = vector.shape_cast %[[FR2]] : vector<64xi8> to vector<8x8xi8>
  %v2t = vector.transpose %v2, [1, 0] : vector<8x8xi8> to vector<8x8xi8>
  return %v1t, %v0t, %v2t : vector<4x16xi8>, vector<16x4xi8>, vector<8x8xi8>
}

// -----

// CHECK-LABEL: transpose_16b_types
// CHECK-SAME: %[[V0:.*]]: vector<4x8xbf16>,
// CHECK-SAME: %[[V1:.*]]: vector<8x4xbf16>,
// CHECK-SAME: %[[V2:.*]]: vector<2x16xi16>,
// CHECK-SAME: %[[V3:.*]]: vector<16x2xi16>
func.func @transpose_16b_types(%v0 : vector<4x8xbf16>,
                               %v1 : vector<8x4xbf16>,
                               %v2 : vector<2x16xi16>,
                               %v3 : vector<16x2xi16>)
    -> (vector<4x8xbf16>, vector<8x4xbf16>, vector<2x16xi16>, vector<16x2xi16>) {
  // CHECK: %[[FV0:.*]] = vector.shape_cast %[[V0]] : vector<4x8xbf16> to vector<32xbf16>
  // CHECK: %[[FR0:.*]] = aievec.shuffle %[[FV0]] [t16_4x8] : vector<32xbf16>
  // CHECK: %[[R0:.*]] = vector.shape_cast %[[FR0]] : vector<32xbf16> to vector<8x4xbf16>
  %v0t = vector.transpose %v0, [1, 0] : vector<4x8xbf16> to vector<8x4xbf16>
  // CHECK: %[[FV1:.*]] = vector.shape_cast %[[V1]] : vector<8x4xbf16> to vector<32xbf16>
  // CHECK: %[[FR1:.*]] = aievec.shuffle %[[FV1]] [t16_8x4] : vector<32xbf16>
  // CHECK: %[[R1:.*]] = vector.shape_cast %[[FR1]] : vector<32xbf16> to vector<4x8xbf16>
  %v1t = vector.transpose %v1, [1, 0] : vector<8x4xbf16> to vector<4x8xbf16>
  // CHECK: %[[FV2:.*]] = vector.shape_cast %[[V2]] : vector<2x16xi16> to vector<32xi16>
  // CHECK: %[[FR2:.*]] = aievec.shuffle %[[FV2]] [t16_2x16] : vector<32xi16>
  // CHECK: %[[R2:.*]] = vector.shape_cast %[[FR2]] : vector<32xi16> to vector<16x2xi16>
  %v2t = vector.transpose %v2, [1, 0] : vector<2x16xi16> to vector<16x2xi16>
  // CHECK: %[[FV3:.*]] = vector.shape_cast %[[V3]] : vector<16x2xi16> to vector<32xi16>
  // CHECK: %[[FR3:.*]] = aievec.shuffle %[[FV3]] [t16_16x2] : vector<32xi16>
  // CHECK: %[[R3:.*]] = vector.shape_cast %[[FR3]] : vector<32xi16> to vector<2x16xi16>
  %v3t = vector.transpose %v3, [1, 0] : vector<16x2xi16> to vector<2x16xi16>
  return %v1t, %v0t, %v3t, %v2t : vector<4x8xbf16>, vector<8x4xbf16>,
                                  vector<2x16xi16>, vector<16x2xi16>
}

// -----

// CHECK-LABEL: transpose_32b_types
// CHECK-SAME: %[[V0:.*]]: vector<4x4xi32>,
// CHECK-SAME: %[[V1:.*]]: vector<4x4xf32>
func.func @transpose_32b_types(%v0 : vector<4x4xi32>,
                               %v1 : vector<4x4xf32>)
    -> (vector<4x4xi32>, vector<4x4xf32>) {
  // CHECK: %[[FV0:.*]] = vector.shape_cast %[[V0]] : vector<4x4xi32> to vector<16xi32>
  // CHECK: %[[FR0:.*]] = aievec.shuffle %[[FV0]] [t32_4x4] : vector<16xi32>
  // CHECK: %[[R0:.*]] = vector.shape_cast %[[FR0]] : vector<16xi32> to vector<4x4xi32>
  %v0t = vector.transpose %v0, [1, 0] : vector<4x4xi32> to vector<4x4xi32>
  // CHECK: %[[FV1:.*]] = vector.shape_cast %[[V1]] : vector<4x4xf32> to vector<16xf32>
  // CHECK: %[[FR1:.*]] = aievec.shuffle %[[FV1]] [t32_4x4] : vector<16xf32>
  // CHECK: %[[R1:.*]] = vector.shape_cast %[[FR1]] : vector<16xf32> to vector<4x4xf32>
  %v1t = vector.transpose %v1, [1, 0] : vector<4x4xf32> to vector<4x4xf32>
  return %v0t, %v1t : vector<4x4xi32>, vector<4x4xf32>
}

// -----

// CHECK-LABEL: transpose_leading_unit_dims
// CHECK-SAME: %[[V0:.*]]: vector<1x1x4x8xbf16>,
// CHECK-SAME: %[[V1:.*]]: vector<1x1x8x4xbf16>,
// CHECK-SAME: %[[V2:.*]]: vector<1x1x1x2x16xi16>,
// CHECK-SAME: %[[V3:.*]]: vector<1x1x1x16x2xi16>
func.func @transpose_leading_unit_dims(%v0 : vector<1x1x4x8xbf16>,
                                       %v1 : vector<1x1x8x4xbf16>,
                                       %v2 : vector<1x1x1x2x16xi16>,
                                       %v3 : vector<1x1x1x16x2xi16>)
    -> (vector<1x1x4x8xbf16>, vector<1x1x8x4xbf16>,
        vector<1x1x1x2x16xi16>, vector<1x1x1x16x2xi16>) {
  // CHECK: %[[FV0:.*]] = vector.shape_cast %[[V0]] : vector<1x1x4x8xbf16> to vector<32xbf16>
  // CHECK: %[[FR0:.*]] = aievec.shuffle %[[FV0]] [t16_4x8] : vector<32xbf16>
  // CHECK: %[[R0:.*]] = vector.shape_cast %[[FR0]] : vector<32xbf16> to vector<1x1x8x4xbf16>
  %v0t = vector.transpose %v0, [0, 1, 3, 2] : vector<1x1x4x8xbf16> to vector<1x1x8x4xbf16>
  // CHECK: %[[FV1:.*]] = vector.shape_cast %[[V1]] : vector<1x1x8x4xbf16> to vector<32xbf16>
  // CHECK: %[[FR1:.*]] = aievec.shuffle %[[FV1]] [t16_8x4] : vector<32xbf16>
  // CHECK: %[[R1:.*]] = vector.shape_cast %[[FR1]] : vector<32xbf16> to vector<1x1x4x8xbf16>
  %v1t = vector.transpose %v1, [0, 1, 3, 2] : vector<1x1x8x4xbf16> to vector<1x1x4x8xbf16>
  // CHECK: %[[FV2:.*]] = vector.shape_cast %[[V2]] : vector<1x1x1x2x16xi16> to vector<32xi16>
  // CHECK: %[[FR2:.*]] = aievec.shuffle %[[FV2]] [t16_2x16] : vector<32xi16>
  // CHECK: %[[R2:.*]] = vector.shape_cast %[[FR2]] : vector<32xi16> to vector<1x1x1x16x2xi16>
  %v2t = vector.transpose %v2, [0, 1, 2, 4, 3] : vector<1x1x1x2x16xi16> to vector<1x1x1x16x2xi16>
  // CHECK: %[[FV3:.*]] = vector.shape_cast %[[V3]] : vector<1x1x1x16x2xi16> to vector<32xi16>
  // CHECK: %[[FR3:.*]] = aievec.shuffle %[[FV3]] [t16_16x2] : vector<32xi16>
  // CHECK: %[[R3:.*]] = vector.shape_cast %[[FR3]] : vector<32xi16> to vector<1x1x1x2x16xi16>
  %v3t = vector.transpose %v3, [0, 1, 2, 4, 3] : vector<1x1x1x16x2xi16> to vector<1x1x1x2x16xi16>
  return %v1t, %v0t, %v3t, %v2t : vector<1x1x4x8xbf16>, vector<1x1x8x4xbf16>,
                                  vector<1x1x1x2x16xi16>, vector<1x1x1x16x2xi16>
}
