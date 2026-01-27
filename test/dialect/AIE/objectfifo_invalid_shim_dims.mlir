//===- objectfifo_invalid_shim_dims.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: `dimensionsToStream` data layout transformations are not supported on shim tile producers

module @objectfifo_invalid_shim_dims {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_0 (%tile20 dimensionsToStream [<size = 1, stride = 1>, <size = 1, stride = 1>], {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<16xi32>
    aie.objectfifo.register_external_buffers @of_0 (%tile20, {%ext_buffer_in}) : (memref<16xi32>)
 }
}

// -----

// CHECK: `dimensionsFromStreamPerConsumer` data layout transformations are not supported on shim tile consumers

module @objectfifo_invalid_shim_dims {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_0 (%tile13, {%tile20 dimensionsFromStream [<size = 1, stride = 1>, <size = 1, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %ext_buffer_out  = aie.external_buffer {sym_name = "ext_buffer_out"}: memref<16xi32>
    aie.objectfifo.register_external_buffers @of_0 (%tile20, {%ext_buffer_out}) : (memref<16xi32>)
 }
}
