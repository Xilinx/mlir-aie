// RUN: aie-opt %s %tosa-to-linalg% | aie-opt %linalg-to-affine% | FileCheck %s

// CHECK-LABEL:func @i32_cast_i16
func.func @i32_cast_i16(%arg0: tensor<1024xi32>) -> tensor<1024xi16> {
  // CHECK: %[[RES:.*]] = arith.trunci %{{.*}} : i32 to i16
  %0 = "tosa.cast"(%arg0) : (tensor<1024xi32>) -> tensor<1024xi16>
  return %0 : tensor<1024xi16>
}

// CHECK-LABEL:func @i32_cast_i8
func.func @i32_cast_i8(%arg0: tensor<1024xi32>) -> tensor<1024xi8> {
  // CHECK: %[[RES:.*]] = arith.trunci %{{.*}} : i32 to i8
  %0 = "tosa.cast"(%arg0) : (tensor<1024xi32>) -> tensor<1024xi8>
  return %0 : tensor<1024xi8>
}

// CHECK-LABEL:func @i32_cast_f32
func.func @i32_cast_f32(%arg0: tensor<1024xi32>) -> tensor<1024xf32> {
  // CHECK: %[[RES:.*]] = arith.sitofp %{{.*}} : i32 to f32
  %0 = "tosa.cast"(%arg0) : (tensor<1024xi32>) -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// CHECK-LABEL:func @i32_cast_f16
func.func @i32_cast_f16(%arg0: tensor<1024xi32>) -> tensor<1024xf16> {
  // CHECK: %[[RES:.*]] = arith.sitofp %{{.*}} : i32 to f16
  %0 = "tosa.cast"(%arg0) : (tensor<1024xi32>) -> tensor<1024xf16>
  return %0 : tensor<1024xf16>
}
