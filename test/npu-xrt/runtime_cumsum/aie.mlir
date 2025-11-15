//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

//┌─────────────────────────────────┐      
//│Input:                           │      
//│[                                │      
//│[1, 1, 1, 1, ..., 1, 1, 1, 1],   │      
//│[0, 0, 0, 0, ..., 0, 0, 0, 0],   │      
//│ .............                   │      
//│]                                │      
//└─────────────────────────────────┘      
//┌───────────────────────────────────────┐
//│Output:                                │
//│[                                      │
//│[1, 1, 1, 1, ..., 1, 1, 1, 1],         │
//│[1, 1, 1, 1, ..., 1, 1, 1, 1],         │
//│[2, 2, 2, 2, ..., 2, 2, 2, 2],         │
//│[4, 4, 4, 4, ..., 4, 4, 4, 4],         │
//│[8, 8, 8, 8, ..., 8, 8, 8, 8],         │
//│[16, 16, 16, 16, ..., 16, 16, 16, 16], │
//│[32, 32, 32, 32, ..., 32, 32, 32, 32], │
//│[64, 64, 64, 64, ..., 64, 64, 64, 64]  │
//│]                                      │
//└───────────────────────────────────────┘

module {
    aie.device(npu1_1col) {

        // AIE Core Function declarations
        func.func private @sum(memref<16xi32>, memref<16xi32>)
        func.func private @zero(memref<16xi32>)

        %shim_noc_tile_0_0 = aie.tile(0, 0)
        %mem_tile_0_1 = aie.tile(0, 1)
        %tile_0_2 = aie.tile(0, 2)

        aie.objectfifo @mem_In(%shim_noc_tile_0_0, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<16xi32>> 
        aie.objectfifo @act_In(%mem_tile_0_1, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<16xi32>> 
        aie.objectfifo.link [@mem_In] -> [@act_In]([] [])
        aie.objectfifo @out(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<16xi32>> 
        aie.objectfifo @mem_out(%mem_tile_0_1, {%shim_noc_tile_0_0}, 1 : i32) : !aie.objectfifo<memref<16xi32>> 
        aie.objectfifo.link [@out] -> [@mem_out]([] [])

        // Buffers used to hold runtime parameters
        %rtp2 = aie.buffer(%tile_0_2) {sym_name = "rtp2"} : memref<16xi32> 

        %core_0_2 = aie.core(%tile_0_2) {
            %c0 = arith.constant 0 : index
            %cmax = arith.constant 294967295 : index
            %c1 = arith.constant 1 : index

            scf.for %arg0 = %c0 to %cmax step %c1 {
                %Out = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elemout = aie.objectfifo.subview.access %Out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

                func.call @zero(%elemout) : (memref<16xi32>) -> ()

                //A simple “do-while” loop can be represented by reducing the “after” block to a simple forwarder.
                scf.while (%arg1 = %c1) : (index) -> (index) {

                    %In = aie.objectfifo.acquire @act_In(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                    %elemin = aie.objectfifo.subview.access %In[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

                    func.call @sum(%elemin, %elemout) : (memref<16xi32>, memref<16xi32>) -> ()

                    aie.objectfifo.release @act_In(Consume, 1)

                    // load the rtp
                    %loopnumber = memref.load %rtp2[%arg0] : memref<16xi32>
                    %loopnumber2index = arith.index_cast %loopnumber : i32 to index

                    // utilize its value to set the condition
                    %next = arith.addi %arg1, %c1 : index
                    %cond = arith.cmpi slt, %arg1, %loopnumber2index : index
                    scf.condition(%cond) %next : index
                } do {
                ^bb0(%arg2: index):
                    scf.yield %arg2: index
                }
                aie.objectfifo.release @out(Produce, 1)
            }
            aie.end
        } {link_with = "sum.o"}

        aie.runtime_sequence @sequence(%xy: memref<128xi32>) {
            aiex.npu.rtp_write(@rtp2, 0, 1)
            // read first row and write to second row
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, metadata = @mem_In} : memref<128xi32>
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 16][1, 1, 1, 16][0, 0, 0, 1]) {id = 2 : i64, metadata = @mem_out} : memref<128xi32>
            aiex.npu.dma_wait {symbol = @mem_out}
            //┌───────────────────────────────┐
            //│Output:                        │
            //│[                              │
            //│[1, 1, 1, 1, ..., 1, 1, 1, 1], │
            //│[1, 1, 1, 1, ..., 1, 1, 1, 1], │
            //│[0, 0, 0, 0, ..., 0, 0, 0, 0], │
            //│............                   │
            //│]                              │
            //└───────────────────────────────┘

            // read first two rows and write to the third row
            aiex.npu.rtp_write(@rtp2, 1, 2)
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 0][1, 1, 2, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @mem_In} : memref<128xi32>
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 32][1, 1, 1, 16][0, 0, 0, 1]) {id = 2 : i64, metadata = @mem_out} : memref<128xi32>
            aiex.npu.dma_wait {symbol = @mem_out}
            //┌───────────────────────────────┐
            //│Output:                        │
            //│[                              │
            //│[1, 1, 1, 1, ..., 1, 1, 1, 1], │
            //│[1, 1, 1, 1, ..., 1, 1, 1, 1], │
            //│[2, 2, 2, 2, ..., 2, 2, 2, 2], │
            //│[0, 0, 0, 0, ..., 0, 0, 0, 0], │
            //│............                   │
            //│]                              │
            //└───────────────────────────────┘


            // read first three rows and write to the 4th row
            aiex.npu.rtp_write(@rtp2, 2, 3)
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 0][1, 1, 3, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @mem_In} : memref<128xi32>
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 48][1, 1, 1, 16][0, 0, 0, 1]) {id = 2 : i64, metadata = @mem_out} : memref<128xi32>
            aiex.npu.dma_wait {symbol = @mem_out}
            aiex.npu.rtp_write(@rtp2, 3, 4)
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 0][1, 1, 4, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @mem_In} : memref<128xi32>
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 64][1, 1, 1, 16][0, 0, 0, 1]) {id = 2 : i64, metadata = @mem_out} : memref<128xi32>
            aiex.npu.dma_wait {symbol = @mem_out}
            aiex.npu.rtp_write(@rtp2, 4, 5)
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 0][1, 1, 5, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @mem_In} : memref<128xi32>
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 80][1, 1, 1, 16][0, 0, 0, 1]) {id = 2 : i64, metadata = @mem_out} : memref<128xi32>
            aiex.npu.dma_wait {symbol = @mem_out}
            aiex.npu.rtp_write(@rtp2, 5, 6)
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 0][1, 1, 6, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @mem_In} : memref<128xi32>
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 96][1, 1, 1, 16][0, 0, 0, 1]) {id = 2 : i64, metadata = @mem_out} : memref<128xi32>
            aiex.npu.dma_wait {symbol = @mem_out}
            aiex.npu.rtp_write(@rtp2, 6, 7)
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 0][1, 1, 7, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @mem_In} : memref<128xi32>
            aiex.npu.dma_memcpy_nd(%xy[0, 0, 0, 112][1, 1, 1, 16][0, 0, 0, 1]) {id = 2 : i64, metadata = @mem_out} : memref<128xi32>
            aiex.npu.dma_wait {symbol = @mem_out}
        }
    }
}

