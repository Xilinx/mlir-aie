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

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

// This test establishes a broadcast through an object FIFO with producer tile (1, 3) and 
// multiple consumer tiles. This broadcast is achieved through tileDMAs independently of
// whether cores share memory or not. As the initial object FIFO is "split", its size is
// adjusted based on the maximum acquire value of the producer.

module @broadcast {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile13 = AIE.tile(1, 3)
        %tile14 = AIE.tile(1, 4)
        %tile33 = AIE.tile(3, 3)

        %buff_out_12 = AIE.buffer(%tile12) { sym_name = "out12" } :  memref<4x16xi32>
        %lock_out_12 = AIE.lock(%tile12, 0) { sym_name = "lock_out12" }
        %buff_out_14 = AIE.buffer(%tile14) { sym_name = "out14" } :  memref<4x16xi32>
        %lock_out_14 = AIE.lock(%tile14, 0) { sym_name = "lock_out14" }
        %buff_out_33 = AIE.buffer(%tile33) { sym_name = "out33" } :  memref<4x16xi32>
        %lock_out_33 = AIE.lock(%tile33, 0) { sym_name = "lock_out33" }

        AIE.objectfifo @objfifo (%tile13, {%tile12, %tile14, %tile33}, 7 : i32) : !AIE.objectfifo<memref<16xi32>>

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

        %core13 = AIE.core(%tile13) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 4 : index
            
            scf.for %indexInLine = %c0 to %height step %c1 {
                %subview0 = AIE.objectfifo.acquire @objfifo (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
                %elem0 = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @generateLineScalar(%elem0) : (memref<16xi32>) -> ()
                AIE.objectfifo.release @objfifo (Produce, 1)
            }
            
            AIE.end
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

        %core12 = AIE.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 4 : index

            AIE.useLock(%lock_out_12, "Acquire", 0)

            scf.for %indexInLine = %c0 to %height step %c1 {
                %subview0 = AIE.objectfifo.acquire @objfifo (Consume, 1) : !AIE.objectfifosubview<memref<16xi32>>
                %elem0 = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @storeLineScalar(%elem0, %indexInLine, %buff_out_12) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
                AIE.objectfifo.release @objfifo (Consume, 1)
            }

            AIE.useLock(%lock_out_12, "Release", 1)
            
            AIE.end
        }

        %core14 = AIE.core(%tile14) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %height = arith.constant 4 : index

            AIE.useLock(%lock_out_14, "Acquire", 0)
            
            scf.for %indexInLine = %c0 to %height step %c2 {
                %subview = AIE.objectfifo.acquire  @objfifo (Consume, 2) : !AIE.objectfifosubview<memref<16xi32>>
                %elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elem1 = AIE.objectfifo.subview.access %subview[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @storeLineScalar(%elem0, %indexInLine, %buff_out_14) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
                %indexPlusOne = arith.addi %indexInLine, %c1 : index
                func.call @storeLineScalar(%elem1, %indexPlusOne, %buff_out_14) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
                AIE.objectfifo.release @objfifo (Consume, 2)
            }
                
            AIE.useLock(%lock_out_14, "Release", 1)
            
            AIE.end
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

        %core33 = AIE.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c3 = arith.constant 3 : index
            %height = arith.constant 3 : index

            AIE.useLock(%lock_out_33, "Acquire", 0)

            scf.for %indexInLine = %c0 to %height step %c1 {
                %subview = AIE.objectfifo.acquire @objfifo (Consume, 2) : !AIE.objectfifosubview<memref<16xi32>>
                %elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elem1 = AIE.objectfifo.subview.access %subview[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @addAndStore(%elem0, %elem1, %indexInLine, %buff_out_33) : (memref<16xi32>, memref<16xi32>, index, memref<4x16xi32>) -> ()
                AIE.objectfifo.release @objfifo (Consume, 1)
            }

            %subview = AIE.objectfifo.acquire @objfifo (Consume, 1) : !AIE.objectfifosubview<memref<16xi32>>
            %elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @storeLineScalar(%elem0, %c3, %buff_out_33) : (memref<16xi32>, index, memref<4x16xi32>) -> ()
            AIE.objectfifo.release @objfifo (Consume, 1)

            AIE.useLock(%lock_out_33, "Release", 1)
            
            AIE.end
        }
    }
}
