//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
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
module @passThroughLine_aie2 {

 	AIE.device(ipu) {
        // declare kernel external kernel function 
        func.func private @passThroughLine(%in: memref<1920xui8>, %out: memref<1920xui8>, %tilewidth: i32) -> ()
        
        // Declare tile object of the AIE class located at position col 1, row 4
        %tile00 = AIE.tile(0, 0)
        %tile02 = AIE.tile(0, 2)

        AIE.objectFifo @inOF(%tile00, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @outOF(%tile02, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
       
        // Define the algorithm for the core of tile(0,2) 
        %core02 = AIE.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %tileheight = arith.constant 1080  : index
            %tilewidth  = arith.constant 1920 : i32
            
            scf.for %iter = %c0 to %tileheight step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @inOF(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemIn = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewOut = AIE.objectFifo.acquire @outOF(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>

                func.call @passThroughLine(%elemIn, %elemOut, %tilewidth) : (memref<1920xui8>, memref<1920xui8>, i32) -> ()

                // Release objectFifos
                AIE.objectFifo.release @inOF(Consume, 1)
                AIE.objectFifo.release @outOF(Produce, 1)
            }
            AIE.end
        } { link_with="passThrough.cc.o" } // indicate kernel object name used by this core

        func.func @sequence(%in : memref<518400xi32>, %arg1 : memref<1xi32>, %out : memref<518400xi32>) {
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %tileheight = arith.constant 1080  : i32
            %tilewidth  = arith.constant 480 : i32  // in 32b words so tileWidth/4

            //dma_memcpy_nd ([offset in 32b words][length in 32b words][stride in 32b words])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in[%c0, %c0, %c0, %c0][%c1, %c1, %tileheight, %tilewidth][%c0, %c0, %tilewidth]) { metadata = @inOF, id = 1 : i32 } : (i32, i32, memref<518400xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0, %c0, %c0, %c0][%c1, %c1, %tileheight, %tilewidth][%c0, %c0, %tilewidth]) { metadata = @outOF, id = 0 : i32 } : (i32, i32, memref<518400xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            return
        }
    }
}
