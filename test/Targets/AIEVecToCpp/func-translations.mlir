//===- func-translations.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// RUN: aie-translate %s -aievec-to-cpp | FileCheck %s


// CHECK: int32_t external_function(v16int32);
func.func private @external_function(%v : vector<16xi32>) -> i32

// CHECK: void external_function_with_memref(int16_t * restrict);
func.func private @external_function_with_memref(%m : memref<64xi16>)

// CHECK: void external_function_with_dynamic_memref(int8_t * restrict, size_t);
func.func private @external_function_with_dynamic_memref(%m : memref<?xi8>)
