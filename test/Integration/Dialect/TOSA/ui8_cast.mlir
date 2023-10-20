// RUN: aie-opt %s %tosa-to-linalg% | aie-opt %linalg-to-affine% | FileCheck %s

// CHECK-LABEL:func @ui8_cast_f32
func.func @ui8_cast_f32(%arg0: tensor<1024xui8>) -> tensor<1024xf32> {
  // CHECK: %[[T1:.*]] = builtin.unrealized_conversion_cast %{{.*}} : ui8 to i8
  // CHECK: %[[RES:.*]] = arith.uitofp %[[T1]] : i8 to f32
  %0 = "tosa.cast"(%arg0) : (tensor<1024xui8>) -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// CHECK-LABEL:func @ui8_cast_f16
func.func @ui8_cast_f16(%arg0: tensor<1024xui8>) -> tensor<1024xf16> {
  // CHECK: %[[T1:.*]] = builtin.unrealized_conversion_cast %{{.*}} : ui8 to i8
  // CHECK: %[[RES:.*]] = arith.uitofp %[[T1]] : i8 to f16
  %0 = "tosa.cast"(%arg0) : (tensor<1024xui8>) -> tensor<1024xf16>
  return %0 : tensor<1024xf16>
}
