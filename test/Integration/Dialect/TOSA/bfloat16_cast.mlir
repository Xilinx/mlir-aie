// RUN: aie-opt %s %tosa-to-linalg% | aie-opt %linalg-to-affine% | FileCheck %s

// CHECK-LABEL:func @f16_cast_f32
func.func @f16_cast_f32(%arg0: tensor<1024xf16>) -> tensor<1024xf32> {
  // CHECK: %[[RES:.*]] = arith.extf %{{.*}} : f16 to f32
  %0 = "tosa.cast"(%arg0) : (tensor<1024xf16>) -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}
