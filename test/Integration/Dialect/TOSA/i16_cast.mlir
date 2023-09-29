// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg))" | mlir-opt --linalg-fuse-elementwise-ops --eliminate-empty-tensors --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs allow-unknown-ops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --drop-equivalent-buffer-results --buffer-results-to-out-params --buffer-deallocation --canonicalize --cse --convert-linalg-to-affine-loops | FileCheck %s

// CHECK-LABEL:func @i16_cast_i8
func.func @i16_cast_i8(%arg0: tensor<1024xi16>) -> tensor<1024xi8> {
  // CHECK: %[[RES:.*]] = arith.trunci %{{.*}} : i16 to i8
  %0 = "tosa.cast"(%arg0) : (tensor<1024xi16>) -> tensor<1024xi8>
  return %0 : tensor<1024xi8>
}

// CHECK-LABEL:func @i16_cast_i32
func.func @i16_cast_i32(%arg0: tensor<1024xi16>) -> tensor<1024xi32> {
  // CHECK: %[[RES:.*]] = arith.extsi %{{.*}} : i16 to i32
  %0 = "tosa.cast"(%arg0) : (tensor<1024xi16>) -> tensor<1024xi32>
  return %0 : tensor<1024xi32>
}

// CHECK-LABEL:func @i16_cast_f32
func.func @i16_cast_f32(%arg0: tensor<1024xi16>) -> tensor<1024xf32> {
  // CHECK: %[[RES:.*]] = arith.sitofp %{{.*}} : i16 to f32
  %0 = "tosa.cast"(%arg0) : (tensor<1024xi16>) -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// CHECK-LABEL:func @i16_cast_f16
func.func @i16_cast_f16(%arg0: tensor<1024xi16>) -> tensor<1024xf16> {
  // CHECK: %[[RES:.*]] = arith.sitofp %{{.*}} : i16 to f16
  %0 = "tosa.cast"(%arg0) : (tensor<1024xi16>) -> tensor<1024xf16>
  return %0 : tensor<1024xf16>
}
