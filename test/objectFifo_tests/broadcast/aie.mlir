//===- broadcast.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: September 15th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

// This test establishes a broadcast through an object FIFO with producer tile (1, 3) and 
// multiple consumer tiles. This broadcast is achieved through tileDMAs independently of
// whether cores share memory or not. As the initial object FIFO is "split", its size is
// adjusted based on the maximum acquire value of the producer.

module @broadcast {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        %tile14 = aie.tile(1, 4)
        %tile33 = aie.tile(3, 3)

        %buff_out_12 = aie.buffer(%tile12) { sym_name = "out12" } :  memref<4x16xi32>
        %lock_out_12 = aie.lock(%tile12, 0) { sym_name = "lock_out12" }
        %buff_out_14 = aie.buffer(%tile14) { sym_name = "out14" } :  memref<4x16xi32>
        %lock_out_14 = aie.lock(%tile14, 0) { sym_name = "lock_out14" }
        %buff_out_33 = aie.buffer(%tile33) { sym_name = "out33" } :  memref<4x16xi32>
        %lock_out_33 = aie.lock(%tile33, 0) { sym_name = "lock_out33" }

        aie.objectfifo @objfifo (%tile13, {%tile12, %tile14, %tile33}, 7 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @generateLineScalar(%lineOut : memref<16xi32>) -> () {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index
            
            scf.for %indexInLine = %c0 to %lineWidth step %c1 {
                %value = arith.index_cast %indexInLine : index to i32
                memref.store %value, %lineOut[%indexInLine] : memref<16xi32>
            }
            return
        }

        %core13 = aie.core(%tile13) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 4 : index
            
            scf.for %indexInLine = %c0 to %height step %c1 {
                %subview0 = aie.objectfifo.acquire @objfifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @generateLineScalar(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @objfifo (Produce, 1)
            }
            
            aie.end
        }

        func.func @storeLineScalar(%lineIn : memref<16xi32>, %row : index, %bufferOut : memref<4x16xi32>) -> () {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index

            scf.for %indexInLine = %c0 to %lineWidth step %c1 {
                %value = memref.load %lineIn[%indexInLine] : memref<16xi32>
                memref.store %value, %bufferOut[%row,%indexInLine] : memref<4x16xi32>
            }
            return
        }

        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 4 : index

            aie.use_lock(%lock_out_12, "Acquire", 0)

            scf.for %indexInLine = %c0 to %height step %c1 {
                %subview0 = aie.objectfifo.acquire @objfifo (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @storeLineScalar(%elem0, %indexInLine, %buff_out_12) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
                aie.objectfifo.release @objfifo (Consume, 1)
            }

            aie.use_lock(%lock_out_12, "Release", 1)
            
            aie.end
        }

        %core14 = aie.core(%tile14) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %height = arith.constant 4 : index

            aie.use_lock(%lock_out_14, "Acquire", 0)
            
            scf.for %indexInLine = %c0 to %height step %c2 {
                %subview = aie.objectfifo.acquire  @objfifo (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @storeLineScalar(%elem0, %indexInLine, %buff_out_14) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
                %indexPlusOne = arith.addi %indexInLine, %c1 : index
                func.call @storeLineScalar(%elem1, %indexPlusOne, %buff_out_14) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
                aie.objectfifo.release @objfifo (Consume, 2)
            }
                
            aie.use_lock(%lock_out_14, "Release", 1)
            
            aie.end
        }

        func.func @addAndStore(%lineIn0 : memref<16xi32>, %lineIn1 : memref<16xi32>, %row : index, %bufferOut : memref<4x16xi32>) -> () {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index

            scf.for %indexInLine = %c0 to %lineWidth step %c1 {
                %value0 = memref.load %lineIn0[%indexInLine] : memref<16xi32>
                %value1 = memref.load %lineIn1[%indexInLine] : memref<16xi32>
                %sum = arith.addi %value0, %value1 : i32
                memref.store %sum, %bufferOut[%row,%indexInLine] : memref<4x16xi32>
            }
            return
        }

        %core33 = aie.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c3 = arith.constant 3 : index
            %height = arith.constant 3 : index

            aie.use_lock(%lock_out_33, "Acquire", 0)

            scf.for %indexInLine = %c0 to %height step %c1 {
                %subview = aie.objectfifo.acquire @objfifo (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @addAndStore(%elem0, %elem1, %indexInLine, %buff_out_33) : (memref<16xi32>, memref<16xi32>, index, memref<4x16xi32>) -> ()
                aie.objectfifo.release @objfifo (Consume, 1)
            }

            %subview = aie.objectfifo.acquire @objfifo (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @storeLineScalar(%elem0, %c3, %buff_out_33) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
            aie.objectfifo.release @objfifo (Consume, 1)

            aie.use_lock(%lock_out_33, "Release", 1)
            
            aie.end
        }
    }
}
