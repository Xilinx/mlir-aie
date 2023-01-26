//===- shimRow_mem_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: January 26th 2023
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @shimRow_mem {
// CHECK: }

module @shimRow_mem {
    %tile71 = AIE.tile(7, 1)
    %tile70 = AIE.tile(7, 0)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile70, {%tile71}, 3) : !AIE.objectFifo<memref<16xi32>>

    %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
    AIE.objectFifo.registerExternalBuffers(%tile70, %objFifo : !AIE.objectFifo<memref<16xi32>>, {%ext_buffer_in}) : (memref<64xi32>)

    func.func @some_work(%a : memref<16xi32>, %b : memref<16xi32>) -> () {
        return
    }

    %core71 = AIE.core(%tile71) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        %subview = AIE.objectFifo.acquire<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem1 = AIE.objectFifo.subview.access %subview[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0, %elem1) : (memref<16xi32>, memref<16xi32>) -> ()
        AIE.objectFifo.release<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        
        AIE.end
    }
}
