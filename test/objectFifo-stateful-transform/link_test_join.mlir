//===- link_test_join.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: June 30th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @link_join {
     
module @link_join {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile21 = AIE.tile(2, 1)
        %tile22 = AIE.tile(2, 2)
        %tile23 = AIE.tile(2, 3)
        %tile33 = AIE.tile(3, 3)

        %objFifo1 = AIE.objectFifo.createObjectFifo(%tile22, {%tile21}, 2 : i32) {sym_name = "link1"} : !AIE.objectFifo<memref<128xi8>>
        %objFifo2 = AIE.objectFifo.createObjectFifo(%tile23, {%tile21}, 2 : i32) {sym_name = "link2"} : !AIE.objectFifo<memref<128xi8>>
        %objFifo3 = AIE.objectFifo.createObjectFifo(%tile33, {%tile21}, 2 : i32) {sym_name = "link3"} : !AIE.objectFifo<memref<128xi8>>
        %objFifo4 = AIE.objectFifo.createObjectFifo(%tile33, {%tile21}, 2 : i32) {sym_name = "link3"} : !AIE.objectFifo<memref<128xi8>>

        %objFifoOut = AIE.objectFifo.createObjectFifo(%tile21, {%tile20}, 2 : i32) {sym_name = "link4"} : !AIE.objectFifo<memref<512xi8>>

        %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<512xi8>
        AIE.objectFifo.registerExternalBuffers(%tile20, %objFifo : !AIE.objectFifo<memref<512xi8>>, {%ext_buffer_in}) : (memref<512xi8>)

        AIE.objectFifo.link({%objFifo1, %objFifo2, %objFifo3, %objFifo4}, {%objFifoOut}) : (!AIE.objectFifo<memref<128xi8>>, !AIE.objectFifo<memref<128xi8>>, !AIE.objectFifo<memref<128xi8>>, !AIE.objectFifo<memref<128xi8>>, !AIE.objectFifo<memref<512xi8>>)
    }
}
