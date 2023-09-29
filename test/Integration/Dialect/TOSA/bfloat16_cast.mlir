// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg))" | mlir-opt --linalg-fuse-elementwise-ops --eliminate-empty-tensors --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs allow-unknown-ops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --drop-equivalent-buffer-results --buffer-results-to-out-params --buffer-deallocation --canonicalize --cse --convert-linalg-to-affine-loops | FileCheck %s

// CHECK-LABEL:func @f16_cast_f32
func.func @f16_cast_f32(%arg0: tensor<1024xf16>) -> tensor<1024xf32> {
  // CHECK: %[[RES:.*]] = arith.extf %{{.*}} : f16 to f32
  %0 = "tosa.cast"(%arg0) : (tensor<1024xf16>) -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}
