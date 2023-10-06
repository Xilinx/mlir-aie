// RUN: aie-opt %s --dynamic-size-no-implicit-broadcast -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_dyn_dim_rewrite_to_const_false(
func.func @test_dyn_dim_rewrite_to_const_false(%arg0: tensor<?xbf16>, %arg1: tensor<?xbf16>) -> (i1, i1) attributes {tosa.no_implicit_broadcast_of_dynamic_sizes} {
    // CHECK-NEXT: arith.constant false
    // CHECK-NOT: tensor.dim
    // CHECK-NOT: arith.cmpi
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c0 : tensor<?xbf16>
    %1 = arith.cmpi eq, %dim_1, %c1 : index
    %dim_2 = tensor.dim %arg1, %c0 : tensor<?xbf16>
    %3 = arith.cmpi eq, %dim_2, %c1 : index
    return %1, %3 : i1, i1
}

// -----

// CHECK-LABEL: func.func @test_do_not_rewrite_no_attribute(
func.func @test_do_not_rewrite_no_attribute(%arg0: tensor<?xbf16>, %arg1: tensor<?xbf16>) -> (i1, i1) {
    // CHECK-NOT: arith.constant false
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: tensor.dim
    // CHECK-NEXT: arith.cmpi
    // CHECK-NEXT: tensor.dim
    // CHECK-NEXT: arith.cmpi
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c0 : tensor<?xbf16>
    %1 = arith.cmpi eq, %dim_1, %c1 : index
    %dim_2 = tensor.dim %arg1, %c0 : tensor<?xbf16>
    %3 = arith.cmpi eq, %dim_2, %c1 : index
    return %1, %3 : i1, i1
}

// -----

// CHECK-LABEL: func.func @test_static_dim_folding_with_attribute_and_no_rewrite(
func.func @test_static_dim_folding_with_attribute_and_no_rewrite(%arg0: tensor<1xbf16>, %arg1: tensor<4xbf16>) -> (i1, i1) attributes {tosa.no_implicit_broadcast_of_dynamic_sizes} {
    // CHECK-NEXT: arith.constant true
    // CHECK-NEXT: arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c0 : tensor<1xbf16>
    %1 = arith.cmpi eq, %dim_1, %c1 : index
    %dim_2 = tensor.dim %arg1, %c0 : tensor<4xbf16>
    %3 = arith.cmpi eq, %dim_2, %c1 : index
    return %1, %3 : i1, i1
}

// -----

// CHECK-LABEL: func.func @test_static_dim_folding_without_attribute(
func.func @test_static_dim_folding_without_attribute(%arg0: tensor<1xbf16>, %arg1: tensor<4xbf16>) -> (i1, i1) {
    // CHECK-NEXT: arith.constant true
    // CHECK-NEXT: arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c0 : tensor<1xbf16>
    %1 = arith.cmpi eq, %dim_1, %c1 : index
    %dim_2 = tensor.dim %arg1, %c0 : tensor<4xbf16>
    %3 = arith.cmpi eq, %dim_2, %c1 : index
    return %1, %3 : i1, i1
}

