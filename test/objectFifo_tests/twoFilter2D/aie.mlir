//===- twoFilter2D.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: January 21th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

// In this test, two objectFifos are created between three tiles; tile (1, 3) is both a
// producer and a consumer. The same function is applied over the data: a filter2D with
// a 3x3 kernel which adds together the values of input lines at the same positions. 
// At the top and bottom borders only two lines are acquired, instead of three.

module @twoFilter2D  {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        %tile14 = aie.tile(1, 4)

        %buff_out = aie.buffer(%tile14) { sym_name = "out" } :  memref<10x16xi32>
        %lock_out = aie.lock(%tile14, 0) { sym_name = "lock_out" }

        aie.objectfifo @of1 (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile13, {%tile14}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        // Kernel Functions
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

        func.func @firstFilterTwoLines(%lineIn0:memref<16xi32>, %lineIn1:memref<16xi32>, %bufferOut:memref<16xi32>) -> () {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index

            scf.for %indexInLine = %c0 to %lineWidth step %c1 {
                %value0 = memref.load %lineIn0[%indexInLine] : memref<16xi32>
                %value1 = memref.load %lineIn1[%indexInLine] : memref<16xi32>
                %sum = arith.addi %value0, %value1 : i32
                memref.store %sum, %bufferOut[%indexInLine] : memref<16xi32>
            }
            return
        }

        func.func @firstFilterThreeLines(%lineIn0:memref<16xi32>, %lineIn1:memref<16xi32>, %lineIn2:memref<16xi32>, %bufferOut:memref<16xi32>) -> () {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index

            scf.for %indexInLine = %c0 to %lineWidth step %c1 {
                %value0 = memref.load %lineIn0[%indexInLine] : memref<16xi32>
                %value1 = memref.load %lineIn1[%indexInLine] : memref<16xi32>
                %value2 = memref.load %lineIn2[%indexInLine] : memref<16xi32>
                %tmpSum = arith.addi %value0, %value1 : i32
                %sum = arith.addi %tmpSum, %value2 : i32
                memref.store %sum, %bufferOut[%indexInLine] : memref<16xi32>
            }
            return
        }

        func.func @secondFilterTwoLines(%lineIn0:memref<16xi32>, %lineIn1:memref<16xi32>, %row:index, %bufferOut:memref<10x16xi32>) -> () {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index

            scf.for %indexInLine = %c0 to %lineWidth step %c1 {
                %value0 = memref.load %lineIn0[%indexInLine] : memref<16xi32>
                %value1 = memref.load %lineIn1[%indexInLine] : memref<16xi32>
                %sum = arith.addi %value0, %value1 : i32
                memref.store %sum, %bufferOut[%row,%indexInLine] : memref<10x16xi32>
            }
            return
        }

        func.func @secondFilterThreeLines(%lineIn0:memref<16xi32>, %lineIn1:memref<16xi32>, %lineIn2:memref<16xi32>, %row:index, %bufferOut:memref<10x16xi32>) -> () {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index

            scf.for %indexInLine = %c0 to %lineWidth step %c1 {
                %value0 = memref.load %lineIn0[%indexInLine] : memref<16xi32>
                %value1 = memref.load %lineIn1[%indexInLine] : memref<16xi32>
                %value2 = memref.load %lineIn2[%indexInLine] : memref<16xi32>
                %tmpSum = arith.addi %value0, %value1 : i32
                %sum = arith.addi %tmpSum, %value2 : i32
                memref.store %sum, %bufferOut[%row,%indexInLine] : memref<10x16xi32>
            }
            return
        }
        
        // Producer of @of1
        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 10 : index

            scf.for %indexInHeight = %c0 to %height step %c1 { 
                %subview = aie.objectfifo.acquire @of1 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @generateLineScalar(%indexInHeight, %elem0) : (index, memref<16xi32>) -> ()
                aie.objectfifo.release @of1 (Produce, 1)
            }
            
            aie.end
        }

        // Consumer of objFifoOne; Producer of @of2
        %core13 = aie.core(%tile13) {
            %zero = arith.constant 0 : i32
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index
            %height = arith.constant 9 : index

            // Top Border
            %subviewOneTop = aie.objectfifo.acquire @of1 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elemOneTop0 = aie.objectfifo.subview.access %subviewOneTop[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elemOneTop1 = aie.objectfifo.subview.access %subviewOneTop[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

            %subviewTwoTop = aie.objectfifo.acquire @of2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elemTwoTop0 = aie.objectfifo.subview.access %subviewTwoTop[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

            func.call @firstFilterTwoLines(%elemOneTop0, %elemOneTop1, %elemTwoTop0) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()

            aie.objectfifo.release @of2 (Produce, 1)

            // Middle 
            scf.for %indexInHeight = %c1 to %height step %c1 { 
                %subviewOne = aie.objectfifo.acquire @of1 (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
                %elemOne0 = aie.objectfifo.subview.access %subviewOne[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elemOne1 = aie.objectfifo.subview.access %subviewOne[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elemOne2 = aie.objectfifo.subview.access %subviewOne[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

                %subviewTwo = aie.objectfifo.acquire @of2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elemTwo0 = aie.objectfifo.subview.access %subviewTwo[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                
                func.call @firstFilterThreeLines(%elemOne0, %elemOne1, %elemOne2, %elemTwo0) : (memref<16xi32>, memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()

                aie.objectfifo.release @of1 (Consume, 1)
                aie.objectfifo.release @of2 (Produce, 1)
            }

            // Bottom Border
            %subviewOneBottom = aie.objectfifo.acquire @of1 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elemOneBottom0 = aie.objectfifo.subview.access %subviewOneBottom[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elemOneBottom1 = aie.objectfifo.subview.access %subviewOneBottom[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

            %subviewTwoBottom = aie.objectfifo.acquire @of2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elemTwoBottom0 = aie.objectfifo.subview.access %subviewTwoBottom[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            
            func.call @firstFilterTwoLines(%elemOneBottom0, %elemOneBottom1, %elemTwoBottom0) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()

            aie.objectfifo.release @of1 (Consume, 2)
            aie.objectfifo.release @of2 (Produce, 1)
            
            aie.end
        }

        // Consumer of objFifoTwo
        %core14 = aie.core(%tile14) {
            %zero = arith.constant 0 : i32
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %lineWidth = arith.constant 16 : index
            %height = arith.constant 9 : index

            aie.use_lock(%lock_out, "Acquire", 0) // acquire output buffer for produce

            // Top Border
            %subviewTop = aie.objectfifo.acquire @of2 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elemTop0 = aie.objectfifo.subview.access %subviewTop[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elemTop1 = aie.objectfifo.subview.access %subviewTop[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @secondFilterTwoLines(%elemTop0, %elemTop1, %c0, %buff_out) : (memref<16xi32>, memref<16xi32>, index, memref<10x16xi32>) -> ()

            // Middle
            scf.for %indexInHeight = %c1 to %height step %c1 { 
                %subview = aie.objectfifo.acquire @of2 (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elem2 = aie.objectfifo.subview.access %subview[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @secondFilterThreeLines(%elem0, %elem1, %elem2, %indexInHeight, %buff_out) : (memref<16xi32>, memref<16xi32>, memref<16xi32>, index, memref<10x16xi32>) -> ()
                aie.objectfifo.release @of2 (Consume, 1)
            }

            // Bottom Border
            %subviewBottom = aie.objectfifo.acquire @of2 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elemBottom0 = aie.objectfifo.subview.access %subviewBottom[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elemBottom1 = aie.objectfifo.subview.access %subviewBottom[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @secondFilterTwoLines(%elemBottom0, %elemBottom1, %height, %buff_out) : (memref<16xi32>, memref<16xi32>, index, memref<10x16xi32>) -> ()
            aie.objectfifo.release @of2 (Consume, 2)

            aie.use_lock(%lock_out, "Release", 1) // release output buffer for consume
            
            aie.end
        }
    }
}
