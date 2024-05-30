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

// -----

func.func @shuffle_i8(%v : vector<64xi8>) -> vector<64xi8> {
  // CHECK: aievec.shuffle %{{.*}} [t8_8x8]
  %0 = aievec.shuffle %v [t8_8x8] : vector<64xi8>
  // CHECK: aievec.shuffle %{{.*}} [t8_16x4]
  %1 = aievec.shuffle %0 [t8_16x4] : vector<64xi8>
  // CHECK: aievec.shuffle %{{.*}} [t8_4x16]
  %2 = aievec.shuffle %1 [t8_4x16] : vector<64xi8>
  // CHECK: aievec.shuffle %{{.*}} [t8_8x4]
  %3 = aievec.shuffle %2 [t8_8x4] : vector<64xi8>
  // CHECK: aievec.shuffle %{{.*}} [t8_4x8]
  %4 = aievec.shuffle %3 [t8_4x8] : vector<64xi8>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_lo]
  %5 = aievec.shuffle %v, %4 [t8_64x2_lo] : vector<64xi8>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_hi]
  %6 = aievec.shuffle %5, %v [t8_64x2_hi] : vector<64xi8>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t8_2x64_lo]
  %7 = aievec.shuffle %v, %6 [t8_2x64_lo] : vector<64xi8>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t8_2x64_hi]
  %8 = aievec.shuffle %7, %v [t8_2x64_hi] : vector<64xi8>
  return %8 : vector<64xi8>
}

// -----

func.func @shuffle_i16(%v : vector<32xi16>) -> vector<32xi16> {
  // CHECK: aievec.shuffle %{{.*}} [t16_8x4]
  %0 = aievec.shuffle %v [t16_8x4] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}} [t16_4x8]
  %1 = aievec.shuffle %0 [t16_4x8] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}} [t16_1x2_flip]
  %2 = aievec.shuffle %1 [t16_1x2_flip] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}} [t16_4x4]
  %3 = aievec.shuffle %2 [t16_4x4] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}} [t16_4x2]
  %4 = aievec.shuffle %3 [t16_4x2] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}} [t16_2x4]
  %5 = aievec.shuffle %4 [t16_2x4] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}} [t16_8x2]
  %6 = aievec.shuffle %5 [t16_8x2] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}} [t16_2x8]
  %7 = aievec.shuffle %6 [t16_2x8] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}} [t16_16x2]
  %8 = aievec.shuffle %7 [t16_16x2] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}} [t16_2x16]
  %9 = aievec.shuffle %8 [t16_2x16] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_32x2_lo]
  %10 = aievec.shuffle %v, %9 [t16_32x2_lo] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_32x2_hi]
  %11 = aievec.shuffle %10, %v [t16_32x2_hi] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_2x32_lo]
  %12 = aievec.shuffle %v, %11 [t16_2x32_lo] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_2x32_hi]
  %13 = aievec.shuffle %12, %v [t16_2x32_hi] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_lo]
  %14 = aievec.shuffle %v, %13 [t16_16x4_lo] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_hi]
  %15 = aievec.shuffle %14, %v [t16_16x4_hi] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_4x16_lo]
  %16 = aievec.shuffle %v, %15 [t16_4x16_lo] : vector<32xi16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_4x16_hi]
  %17 = aievec.shuffle %16, %v [t16_4x16_hi] : vector<32xi16>
  return %17 : vector<32xi16>
}

// -----

func.func @shuffle_bf16(%v : vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: aievec.shuffle %{{.*}} [t16_8x4]
  %0 = aievec.shuffle %v [t16_8x4] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}} [t16_4x8]
  %1 = aievec.shuffle %0 [t16_4x8] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}} [t16_1x2_flip]
  %2 = aievec.shuffle %1 [t16_1x2_flip] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}} [t16_4x4]
  %3 = aievec.shuffle %2 [t16_4x4] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}} [t16_4x2]
  %4 = aievec.shuffle %3 [t16_4x2] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}} [t16_2x4]
  %5 = aievec.shuffle %4 [t16_2x4] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}} [t16_8x2]
  %6 = aievec.shuffle %5 [t16_8x2] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}} [t16_2x8]
  %7 = aievec.shuffle %6 [t16_2x8] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}} [t16_16x2]
  %8 = aievec.shuffle %7 [t16_16x2] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}} [t16_2x16]
  %9 = aievec.shuffle %8 [t16_2x16] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_32x2_lo]
  %10 = aievec.shuffle %v, %9 [t16_32x2_lo] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_32x2_hi]
  %11 = aievec.shuffle %10, %v [t16_32x2_hi] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_2x32_lo]
  %12 = aievec.shuffle %v, %11 [t16_2x32_lo] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_2x32_hi]
  %13 = aievec.shuffle %12, %v [t16_2x32_hi] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_lo]
  %14 = aievec.shuffle %v, %13 [t16_16x4_lo] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_hi]
  %15 = aievec.shuffle %14, %v [t16_16x4_hi] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_4x16_lo]
  %16 = aievec.shuffle %v, %15 [t16_4x16_lo] : vector<32xbf16>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t16_4x16_hi]
  %17 = aievec.shuffle %16, %v [t16_4x16_hi] : vector<32xbf16>
  return %17 : vector<32xbf16>
}

// -----

func.func @shuffle_i32(%v : vector<16xi32>) -> vector<16xi32> {
  // CHECK: aievec.shuffle %{{.*}} [t32_4x4]
  %0 = aievec.shuffle %v, [t32_4x4] : vector<16xi32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo]
  %1 = aievec.shuffle %0, %v [t32_16x2_lo] : vector<16xi32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi]
  %2 = aievec.shuffle %v, %1 [t32_16x2_hi] : vector<16xi32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo]
  %3 = aievec.shuffle %2, %v [t32_2x16_lo] : vector<16xi32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi]
  %4 = aievec.shuffle %v, %3 [t32_2x16_hi] : vector<16xi32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_8x4_lo]
  %5 = aievec.shuffle %4, %v [t32_8x4_lo] : vector<16xi32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_8x4_hi]
  %6 = aievec.shuffle %v, %5 [t32_8x4_hi] : vector<16xi32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_4x8_lo]
  %7 = aievec.shuffle %6, %v [t32_4x8_lo] : vector<16xi32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_4x8_hi]
  %8 = aievec.shuffle %v, %7 [t32_4x8_hi] : vector<16xi32>
  return %8 : vector<16xi32>
}

// -----

func.func @shuffle_f32(%v : vector<16xf32>) -> vector<16xf32> {
  // CHECK: aievec.shuffle %{{.*}} [t32_4x4]
  %0 = aievec.shuffle %v, [t32_4x4] : vector<16xf32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo]
  %1 = aievec.shuffle %0, %v [t32_16x2_lo] : vector<16xf32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi]
  %2 = aievec.shuffle %v, %1 [t32_16x2_hi] : vector<16xf32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo]
  %3 = aievec.shuffle %2, %v [t32_2x16_lo] : vector<16xf32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi]
  %4 = aievec.shuffle %v, %3 [t32_2x16_hi] : vector<16xf32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_8x4_lo]
  %5 = aievec.shuffle %4, %v [t32_8x4_lo] : vector<16xf32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_8x4_hi]
  %6 = aievec.shuffle %v, %5 [t32_8x4_hi] : vector<16xf32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_4x8_lo]
  %7 = aievec.shuffle %6, %v [t32_4x8_lo] : vector<16xf32>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t32_4x8_hi]
  %8 = aievec.shuffle %v, %7 [t32_4x8_hi] : vector<16xf32>
  return %8 : vector<16xf32>
}

// -----

func.func @shuffle_i64(%v : vector<8xi64>) -> vector<8xi64> {
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo]
  %0 = aievec.shuffle %v, %v [t64_8x2_lo] : vector<8xi64>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_hi]
  %1 = aievec.shuffle %0, %v [t64_8x2_hi] : vector<8xi64>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t64_2x8_lo]
  %2 = aievec.shuffle %v, %1 [t64_2x8_lo] : vector<8xi64>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t64_2x8_hi]
  %3 = aievec.shuffle %2, %v [t64_2x8_hi] : vector<8xi64>
  return %3 : vector<8xi64>
}

// -----

func.func @shuffle_i128(%v : vector<4xi128>) -> vector<4xi128> {
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_lo]
  %0 = aievec.shuffle %v, %v [t128_4x2_lo] : vector<4xi128>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_hi]
  %1 = aievec.shuffle %0, %v [t128_4x2_hi] : vector<4xi128>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t128_2x4_lo]
  %2 = aievec.shuffle %v, %1 [t128_2x4_lo] : vector<4xi128>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t128_2x4_hi]
  %3 = aievec.shuffle %2, %v [t128_2x4_hi] : vector<4xi128>
  return %3 : vector<4xi128>
}

// -----

func.func @shuffle_i256(%v : vector<2xi256>) -> vector<2xi256> {
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t256_2x2_lo]
  %0 = aievec.shuffle %v, %v [t256_2x2_lo] : vector<2xi256>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t256_2x2_hi]
  %1 = aievec.shuffle %0, %v [t256_2x2_hi] : vector<2xi256>
  return %1 : vector<2xi256>
}

// -----

func.func @shuffle_i512(%v : vector<1xi512>) -> vector<1xi512> {
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t512_1x2_lo]
  %0 = aievec.shuffle %v, %v [t512_1x2_lo] : vector<1xi512>
  // CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t512_1x2_hi]
  %1 = aievec.shuffle %0, %v [t512_1x2_hi] : vector<1xi512>
  return %1 : vector<1xi512>
}
