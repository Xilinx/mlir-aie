//===- base_test_2.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: September 9th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @tileDMAChannelGeneration {

module @tileDMAChannelGeneration {
    %tile12 = AIE.tile(1, 2)
    %tile33 = AIE.tile(3, 3)

    %objFifo0 = AIE.objectFifo.createObjectFifo(%tile12, {%tile33}, 4) : !AIE.objectFifo<memref<16xi32>>
    %objFifo1 = AIE.objectFifo.createObjectFifo(%tile12, {%tile33}, 2) : !AIE.objectFifo<memref<16xi32>>

    %objFifo2 = AIE.objectFifo.createObjectFifo(%tile33, {%tile12}, 4) : !AIE.objectFifo<memref<16xi32>>
    %objFifo3 = AIE.objectFifo.createObjectFifo(%tile33, {%tile12}, 2) : !AIE.objectFifo<memref<16xi32>>

    func.func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }

    %core12 = AIE.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview0 = AIE.objectFifo.acquire<Produce>(%objFifo0 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %subview1 = AIE.objectFifo.acquire<Produce>(%objFifo1 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>

            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()

            AIE.objectFifo.release<Produce>(%objFifo0 : !AIE.objectFifo<memref<16xi32>>, 1)
            AIE.objectFifo.release<Produce>(%objFifo1 : !AIE.objectFifo<memref<16xi32>>, 1)
        }

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview2 = AIE.objectFifo.acquire<Consume>(%objFifo2 : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
            %subview3 = AIE.objectFifo.acquire<Consume>(%objFifo3 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem20 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem21 = AIE.objectFifo.subview.access %subview2[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem3 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>

            func.call @some_work(%elem20) : (memref<16xi32>) -> ()
            func.call @some_work(%elem21) : (memref<16xi32>) -> ()
            func.call @some_work(%elem3) : (memref<16xi32>) -> ()

            AIE.objectFifo.release<Consume>(%objFifo2 : !AIE.objectFifo<memref<16xi32>>, 1)
            AIE.objectFifo.release<Consume>(%objFifo3 : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }
}