//===- hello_world.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Declare this MLIR module. A block encapsulates all 
// AIE tiles, buffers, and communication in an AI Engine design
module @printf {

AIE.device(ipu) {
        // declare kernel external kernel function 
        func.func private @kernel(%in: memref<512xui32>, %out: memref<512xui32>, %logout: memref<512xui32>) -> ()
        
        // Declare tile object of the AIE class located at position col 1, row 4
        %tile00 = AIE.tile(0, 0)
        %tile02 = AIE.tile(0, 4)

        AIE.objectfifo @inOF(%tile00, {%tile02}, 2 : i32) : !AIE.objectfifo<memref<512xui32>>
        AIE.objectfifo @outOF(%tile02, {%tile00}, 2 : i32) : !AIE.objectfifo<memref<512xui32>>
        AIE.objectfifo @logoutOF(%tile02, {%tile00}, 2 : i32) : !AIE.objectfifo<memref<512xui32>>
       
        // Define the algorithm for the core of tile(0,2) 
        %core02 = AIE.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index

            // Acquire objectFifos and get subviews
            %subviewIn = AIE.objectfifo.acquire @inOF(Consume, 1) : !AIE.objectfifosubview<memref<512xui32>>
            %elemIn = AIE.objectfifo.subview.access %subviewIn[0] : !AIE.objectfifosubview<memref<512xui32>> -> memref<512xui32>
            %subviewOut = AIE.objectfifo.acquire @outOF(Produce, 1) : !AIE.objectfifosubview<memref<512xui32>>
            %elemOut = AIE.objectfifo.subview.access %subviewOut[0] : !AIE.objectfifosubview<memref<512xui32>> -> memref<512xui32>
            %subviewLogOut = AIE.objectfifo.acquire @logoutOF(Produce, 1) : !AIE.objectfifosubview<memref<512xui32>>
            %elemLogout = AIE.objectfifo.subview.access %subviewLogOut[0] : !AIE.objectfifosubview<memref<512xui32>> -> memref<512xui32>

            func.call @kernel(%elemIn, %elemOut, %elemLogout) : (memref<512xui32>, memref<512xui32>, memref<512xui32>) -> ()

            // Release objectFifos
            AIE.objectfifo.release @inOF(Consume, 1)
            AIE.objectfifo.release @outOF(Produce, 1)
            AIE.objectfifo.release @logoutOF(Produce, 1)
           
            AIE.end
        } { link_with="kernel.o" } // indicate kernel object name used by this core

        func.func @sequence(%in : memref<512xui32>, %out : memref<512xui32>, %logout :memref<512xui32>) {
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %tilewidth  = arith.constant 512 : i32  // in 32b words so tileWidth/4

            //dma_memcpy_nd ([offset in 32b words][length in 32b words][stride in 32b words])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %tilewidth][%c0, %c0, %tilewidth]) { metadata = @inOF, id = 1 : i32 } : (i32, i32, memref<512xui32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %tilewidth][%c0, %c0, %tilewidth]) { metadata = @outOF, id = 0 : i32 } : (i32, i32, memref<512xui32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %logout[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %tilewidth][%c0, %c0, %tilewidth]) { metadata = @logoutOF, id = 0 : i32 } : (i32, i32, memref<512xui32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            return
        }
    }
}
