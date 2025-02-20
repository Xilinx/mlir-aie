//===- objectfifo_register_external_buffers_bad.mlir ------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: 'aie.objectfifo.register_external_buffers' op failed to verify that op exists in a shim tile with NOC connection

module @objectfifo_register_external_buffers_bad {
 aie.device(xcve2302) {
    %tile21 = aie.tile(2, 1)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_0 (%tile21, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<16xi32>
    aie.objectfifo.register_external_buffers @of_0 (%tile21, {%ext_buffer_in}) : (memref<16xi32>)
 }
}
