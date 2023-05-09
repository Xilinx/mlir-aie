//===- memTile_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Xilinx Inc.
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 9th 2023
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @memTile {

module @memTile {
 AIE.device(xcve2302) {
    %tile10 = AIE.tile(2, 0)
    %tile11 = AIE.tile(2, 1)
    %tile12 = AIE.tile(2, 2)

    %objFifo0 = AIE.objectFifo.createObjectFifo(%tile10, {%tile11}, 2) : !AIE.objectFifo<memref<16xi32>>

    %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
    AIE.objectFifo.registerExternalBuffers(%tile10, %objFifo0 : !AIE.objectFifo<memref<16xi32>>, {%ext_buffer_in}) : (memref<64xi32>)

    %objFifo1 = AIE.objectFifo.createObjectFifo(%tile11, {%tile12}, 2) : !AIE.objectFifo<memref<16xi32>>
 }
}
