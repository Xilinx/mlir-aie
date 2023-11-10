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
module @edgeDetect_aie2 {

 	AIE.device(ipu) {
        // declare kernel external kernel function 
        func.func private @rgba2grayLine(%in: memref<5120xui8>, %out: memref<1280xui8>, %tileWidth: i32) -> ()
        func.func private @filter2dLine(%lineIn1: memref<1280xui8>, %lineIn2: memref<1280xui8>, %lineIn3: memref<1280xui8>, %out: memref<1280xui8>, %lineWidth: i32, %kernel: memref<3x3xi16>) -> ()
        func.func private @thresholdLine(%in: memref<1280xui8>, %out: memref<1280xui8>, %lineWidth: i32,  %thresholdValue: i16, %maxValue: i16, %thresholdType: i8) -> ()
        func.func private @gray2rgbaLine(%in: memref<1280xui8>, %out: memref<5120xui8>, %tileWidth: i32) -> ()
        func.func private @addWeightedLine(%in1: memref<5120xui8>, %in1: memref<5120xui8>, %out: memref<5120xui8>, %tileWidth: i32, %alpha: i16, %beta: i16, %gamma: i8) -> ()

        // Declare tile object of the AIE class located at position col 1, row 4
        %tile00 = AIE.tile(0, 0)
        %tile01 = AIE.tile(0, 1)
        %tile02 = AIE.tile(0, 2)
        %tile03 = AIE.tile(0, 3)
        %tile04 = AIE.tile(0, 4)
        %tile05 = AIE.tile(0, 5)

        // Run-time parameters
        %rtp = AIE.buffer(%tile05) {sym_name = "rtp"} : memref<16xi32>

        // Declare in and out object FIFOs
        AIE.objectFifo @inOF_L3L2(%tile00, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<5120xui8>>
        AIE.objectFifo @inOF_L2L1(%tile01, {%tile02,%tile05}, [2,2,7]) : !AIE.objectFifo<memref<5120xui8>>
        AIE.objectFifo.link [@inOF_L3L2] -> [@inOF_L2L1] ()
        
        AIE.objectFifo @outOFL2L3(%tile01, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<5120xui8>>
        AIE.objectFifo @outOFL1L2(%tile05, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<5120xui8>>
        AIE.objectFifo.link [@outOFL1L2] -> [@outOFL2L3] ()

        // Declare task to task object FIFOs

        AIE.objectFifo @OF_2to3(%tile02, {%tile03}, 4 : i32) : !AIE.objectFifo<memref<1280xui8>>
        AIE.objectFifo @OF_3to4(%tile03, {%tile04}, 2 : i32) : !AIE.objectFifo<memref<1280xui8>>
        AIE.objectFifo @OF_4to5(%tile04, {%tile05}, 2 : i32) : !AIE.objectFifo<memref<1280xui8>>
        AIE.objectFifo @OF_5to5(%tile05, {%tile05}, 1 : i32) : !AIE.objectFifo<memref<5120xui8>>
       
        // Define the algorithm for the core of tile(0,2) 
        %core02 = AIE.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %tileHeight = arith.constant 720  : index
            %tileWidth  = arith.constant 1280 : i32
            %intmax = arith.constant 0xFFFFFFFF : index
            
            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @inOF_L2L1(Consume, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemIn = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_2to3(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>

                func.call @rgba2grayLine(%elemIn, %elemOut, %tileWidth) : (memref<5120xui8>, memref<1280xui8>, i32) -> ()

                // Release objectFifos
                AIE.objectFifo.release @inOF_L2L1(Consume, 1)
                AIE.objectFifo.release @OF_2to3(Produce, 1)
            }
            AIE.end
        } { link_with="rgba2gray.cc.o" } // indicate kernel object name used by this core

        // Define the algorithm for the core of tile(0,3) 
        %core03 = AIE.core(%tile03) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %tileHeight = arith.constant 720  : index
            %tileHeightMinus1 = arith.constant 719  : index
            %tileWidth  = arith.constant 1280 : i32
            %kernel = memref.alloc() : memref<3x3xi16>

            %kernelValue0 = arith.constant 0 : i16 // 0 * 2^12
            %kernelValue1 = arith.constant 4096 : i16 // 1 * 2^12
            %kernelValueMinus4 = arith.constant -16384 : i16 // -4 * 2^12
            memref.store %kernelValue0, %kernel[%c0, %c0] : memref<3x3xi16>
            memref.store %kernelValue1, %kernel[%c0, %c1] : memref<3x3xi16>
            memref.store %kernelValue0, %kernel[%c0, %c2] : memref<3x3xi16>
            memref.store %kernelValue1, %kernel[%c1, %c0] : memref<3x3xi16>
            memref.store %kernelValueMinus4, %kernel[%c1, %c1] : memref<3x3xi16>
            memref.store %kernelValue1, %kernel[%c1, %c2] : memref<3x3xi16>
            memref.store %kernelValue0, %kernel[%c2, %c0] : memref<3x3xi16>
            memref.store %kernelValue1, %kernel[%c2, %c1] : memref<3x3xi16>
            memref.store %kernelValue0, %kernel[%c2, %c2] : memref<3x3xi16>

            %intmax = arith.constant 0xFFFFFFFF : index
            
            scf.for %reps = %c0 to %intmax step %c1 {
                
                // Preamble : Top Border
                %subviewInPre = AIE.objectFifo.acquire @OF_2to3(Consume, 2) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemPre0 = AIE.objectFifo.subview.access %subviewInPre[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %elemPre1 = AIE.objectFifo.subview.access %subviewInPre[1] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOutPre = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemPreOut = AIE.objectFifo.subview.access %subviewOutPre[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                func.call @filter2dLine(%elemPre0, %elemPre0, %elemPre1, %elemPreOut, %tileWidth, %kernel) : (memref<1280xui8>, memref<1280xui8>, memref<1280xui8>, memref<1280xui8>, i32, memref<3x3xi16>) -> () 
                AIE.objectFifo.release @OF_3to4(Produce, 1)

                // Steady State : Middle
                scf.for %arg3 = %c1 to %tileHeightMinus1 step %c1 {
                %subviewIn = AIE.objectFifo.acquire @OF_2to3(Consume, 3) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elem0 = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %elem1 = AIE.objectFifo.subview.access %subviewIn[1] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %elem2 = AIE.objectFifo.subview.access %subviewIn[2] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                func.call @filter2dLine(%elem0, %elem1, %elem2, %elemOut, %tileWidth, %kernel) : (memref<1280xui8>, memref<1280xui8>, memref<1280xui8>, memref<1280xui8>, i32, memref<3x3xi16>) -> ()
                AIE.objectFifo.release @OF_2to3(Consume, 1)
                AIE.objectFifo.release @OF_3to4(Produce, 1)
                }

                // Postamble : Bottom Border
                %subviewInPost = AIE.objectFifo.acquire @OF_2to3(Consume, 2) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemPost0 = AIE.objectFifo.subview.access %subviewInPost[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %elemPost1 = AIE.objectFifo.subview.access %subviewInPost[1] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOutPost = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemPostOut = AIE.objectFifo.subview.access %subviewOutPost[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                func.call @filter2dLine(%elemPost0, %elemPost1, %elemPost1, %elemPostOut, %tileWidth, %kernel) : (memref<1280xui8>, memref<1280xui8>, memref<1280xui8>, memref<1280xui8>, i32, memref<3x3xi16>) -> ()
                AIE.objectFifo.release @OF_2to3(Consume, 2)
                AIE.objectFifo.release @OF_3to4(Produce, 1)
                
            }

            AIE.end
        } { link_with="filter2d.cc.o" } // indicate kernel object name used by this core


        // Define the algorithm for the core of tile(0,4) 
        %core04 = AIE.core(%tile04) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %tileHeight = arith.constant 720  : index
            %tileWidth  = arith.constant 1280 : i32
            %thresholdValue  = arith.constant 10 : i16
            %thresholdType  = arith.constant 0 : i8
            %maxValue        = arith.constant 255 : i16
            %intmax = arith.constant 0xFFFFFFFF : index
            
            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @OF_3to4(Consume, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemIn = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_4to5(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>

                func.call @thresholdLine(%elemIn, %elemOut, %tileWidth, %thresholdValue, %maxValue, %thresholdType) : (memref<1280xui8>, memref<1280xui8>, i32, i16, i16, i8) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_3to4(Consume, 1)
                AIE.objectFifo.release @OF_4to5(Produce, 1)
            }
            AIE.end
        } { link_with="threshold.cc.o" } // indicate kernel object name used by this core

        // Define the algorithm for the core of tile(0,5) 
        %core05 = AIE.core(%tile05) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %tileHeight = arith.constant 720  : index
            %tileWidth  = arith.constant 1280 : i32
            %tileWidthRGBA = arith.constant 5120 : i32
            
            %intmax = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @OF_4to5(Consume, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemIn = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_5to5(Produce, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>
                func.call @gray2rgbaLine(%elemIn, %elemOut, %tileWidth) : (memref<1280xui8>, memref<5120xui8>, i32) -> ()

                 // Release objectFifos
                AIE.objectFifo.release @OF_4to5(Consume, 1)
                AIE.objectFifo.release @OF_5to5(Produce, 1)
                
                // 2 kernel
                // Acquire objectFifos and get subviews
                %subviewIn1 = AIE.objectFifo.acquire @OF_5to5(Consume, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemIn1 = AIE.objectFifo.subview.access %subviewIn1[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>
                %subviewIn2 = AIE.objectFifo.acquire @inOF_L2L1(Consume, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemIn2 = AIE.objectFifo.subview.access %subviewIn2[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>
                %subviewOut2 = AIE.objectFifo.acquire @outOFL1L2(Produce, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemOut2 = AIE.objectFifo.subview.access %subviewOut2[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>

                %a     = memref.load %rtp[%c0] : memref<16xi32>
                %b     = memref.load %rtp[%c1] : memref<16xi32>
                %g     = memref.load %rtp[%c2] : memref<16xi32>
                %alpha = arith.trunci       %a : i32 to i16
                %beta  = arith.trunci       %b : i32 to i16
                %gamma = arith.trunci       %g : i32 to  i8

                func.call @addWeightedLine(%elemIn1, %elemIn2, %elemOut2, %tileWidthRGBA, %alpha, %beta, %gamma) : (memref<5120xui8>, memref<5120xui8>, memref<5120xui8>, i32, i16, i16, i8) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_5to5(Consume, 1)
                AIE.objectFifo.release @inOF_L2L1(Consume, 1)
                AIE.objectFifo.release @outOFL1L2(Produce, 1)
            }
            AIE.end
        } { link_with="combined_gray2rgba_addWeighted.a" } // indicate kernel object name used by this core

        func.func @sequence(%in : memref<921600xi32>, %arg1 : memref<16x16xi32>, %out : memref<921600xi32>) {
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %tileHeight   = arith.constant 720  : i32
            %tileWidth    = arith.constant 1280 : i32  // in 32b words so tileWidth
            %totalLenRGBA = arith.constant 921600 : i32

            AIEX.ipu.rtp_write(0, 5, 0, 16384) { buffer_sym_name = "rtp" }  // alpha = 1.0
            AIEX.ipu.rtp_write(0, 5, 1, 16384) { buffer_sym_name = "rtp" }  // beta  = 1.0
            AIEX.ipu.rtp_write(0, 5, 2,     0) { buffer_sym_name = "rtp" }  // gamma =   0

            //dma_memcpy_nd ([offset in 32b words][length in 32b words][stride in 32b words])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in [%c0, %c0, %c0, %c0][%c1, %c1, %c1, %totalLenRGBA][%c0, %c0, %c0]) { metadata = @inOF_L3L2, id = 1 : i32 }  : (i32, i32, memref<921600xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %totalLenRGBA][%c0, %c0, %c0]) { metadata = @outOFL2L3, id = 0 : i32 } : (i32, i32, memref<921600xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            return
        }
    }
}
