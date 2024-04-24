//===- ping_pong.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: February 11th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

// This test uses a special case of the objectFifo, an objectFifo with two elements. 
// In this case, the objectFifo is not a circular buffer but a ping-pong one.
// This example is based on an objectFifo between two non-adjacent tiles and shows
// that a flow and dma operations are established between the memory modules.

module @ping_pong {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)

        %buff_out = aie.buffer(%tile33) { sym_name = "out" } :  memref<10x16xi32>
        %lock_out = aie.lock(%tile33, 0) { sym_name = "lock_out" }

        aie.objectfifo @objfifo (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        // Fills the given memref with the same input index value.
        func.func @generateLineScalar(%valueIndex : index, %lineOut : memref<16xi32>) -> () {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index
            
            %value = arith.index_cast %valueIndex : index to i32
            scf.for %indexInLine = %c0 to %lineWidth step %c1 {
                %i = arith.index_cast %indexInLine : index to i32
                %sum = arith.addi %value, %i : i32
                memref.store %sum, %lineOut[%indexInLine] : memref<16xi32>
            }
            return
        }

        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 10 : index

            scf.for %indexInHeight = %c0 to %height step %c1 { 
                // acquire next element for produce
                %subview = aie.objectfifo.acquire @objfifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

                // call generator function
                func.call @generateLineScalar(%indexInHeight, %elem0) : (index, memref<16xi32>) -> ()

                // release next element for consume
                aie.objectfifo.release @objfifo (Produce, 1)
            }
            
            aie.end
        }

        // Stores the given memref in the bufferOut at the given row index.
        func.func @storeLineScalar(%lineIn : memref<16xi32>, %row : index, %bufferOut : memref<10x16xi32>) -> () {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index

            scf.for %indexInLine = %c0 to %lineWidth step %c1 {
                %value = memref.load %lineIn[%indexInLine] : memref<16xi32>
                memref.store %value, %bufferOut[%row,%indexInLine] : memref<10x16xi32>
            }
            return
        }

        %core33 = aie.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 10 : index

            // acquire output buffer
            aie.use_lock(%lock_out, "Acquire", 0) // acquire for produce

            scf.for %indexInHeight = %c0 to %height step %c1 { 
                // acquire next element for consume
                %subview = aie.objectfifo.acquire @objfifo (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

                // call consumer function
                func.call @storeLineScalar(%elem0, %indexInHeight, %buff_out) : (memref<16xi32>, index, memref<10x16xi32>) -> ()

                // release next element for produce
                aie.objectfifo.release @objfifo (Consume, 1)
            }

            // release output buffer
            aie.use_lock(%lock_out, "Release", 1) // release for consume
            
            aie.end
        }
    }
}
