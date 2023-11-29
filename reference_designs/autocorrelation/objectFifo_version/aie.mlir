//===- aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf 

module @autocorrelation {
    %tile0_1 = AIE.tile(2, 1)
    %tile0_2 = AIE.tile(2, 2)
    %tile0_3 = AIE.tile(2, 3)
    %tile0_4 = AIE.tile(2, 4)
    %tile0_5 = AIE.tile(2, 5)
    %tile0_6 = AIE.tile(2, 6)
    %tile0_7 = AIE.tile(2, 7)
    %tile0_8 = AIE.tile(2, 8)

    %tile7_0 = AIE.tile(7, 0)

    %inputExt = AIE.external_buffer {sym_name = "input"} : memref<1024 x i32>
    %outputExt = AIE.external_buffer {sym_name = "output"} : memref<1024 x i32>

    AIE.objectfifo @of_in (%tile7_0, {%tile0_1, %tile0_2, %tile0_3, %tile0_4}, 1 : i32) : !AIE.objectfifo<memref<1024xi32>>
    AIE.objectfifo @of_out (%tile0_1, {%tile7_0}, 1 : i32) : !AIE.objectfifo<memref<1024xi32>>

    AIE.objectfifo.register_external_buffers(%tile7_0, %objFifoIn : !AIE.objectfifo<memref<1024xi32>>, {%inputExt}) : (memref<1024xi32>)
    AIE.objectfifo.register_external_buffers(%tile7_0, %objFifoOut : !AIE.objectfifo<memref<1024xi32>>, {%outputExt}) : (memref<1024xi32>)

    AIE.objectfifo @of_04_03 (%tile0_4, {%tile0_3}, 1 : i32) : !AIE.objectfifo<memref<1024xi32>>
    AIE.objectfifo @of_03_02 (%tile0_3, {%tile0_2}, 1 : i32) : !AIE.objectfifo<memref<1024xi32>>
    AIE.objectfifo @of_02_01 (%tile0_2, {%tile0_1}, 1 : i32) : !AIE.objectfifo<memref<1024xi32>>

    func.func @autocorrelate(%bufin: memref<1024xi32>, %bufout:memref<1024xi32>, %offset:index, %blocksize:index) -> () {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cend = arith.constant 1024 : index
        %zero = arith.constant 0 : i32

        affine.for %arg1 = %c0 to %blocksize {
            %acc = affine.for %arg0 = %c0 to %cend
                iter_args(%acc_iter = %zero) -> (i32) {
                %a = affine.load %bufin[%arg0 + %arg1 + %offset] : memref<1024xi32>
                %b = affine.load %bufin[%arg0] : memref<1024xi32>
                %product = arith.muli %a, %b: i32
                %sum = arith.addi %acc_iter, %product: i32
                affine.yield %sum : i32
            }
            affine.store %acc, %bufout[%arg1] : memref<1024xi32>
        }
        return
    }

    AIE.core(%tile0_1) {
        %subviewIn = AIE.objectfifo.acquire @of_in (Consume, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        %subviewOut = AIE.objectfifo.acquire @of_out (Produce, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        
        %input = AIE.objectfifo.subview.access %subviewIn[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %output = AIE.objectfifo.subview.access %subviewOut[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %offset = arith.constant 0 : index
        %blocksize = arith.constant 16 : index

        func.call @autocorrelate(%input, %output, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()

        AIE.objectfifo.release @of_in (Consume, 1)

        // Append the prior results, block of 16.
        %subviewIn2 = AIE.objectfifo.acquire @of_in (%objFifo_02_01 : !AIE.objectfifo<memref<1024xi32>>, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        %input2 = AIE.objectfifo.subview.access %subviewIn2[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

        %1 = memref.subview %input2[0][64][1] : memref<1024xi32> to memref<64xi32>
        %2 = memref.subview %output[%blocksize][64][1] : memref<1024xi32> to memref<64xi32, strided<[1], offset: ?>>
        memref.copy %1, %2 : memref<64xi32> to memref<64xi32, strided<[1], offset: ?>>
        
        AIE.objectfifo.release<Consume>(%objFifo_02_01 : !AIE.objectfifo<memref<1024xi32>>, 1)

        AIE.objectfifo.release @of_out (Produce, 1)
        
        AIE.end
    }

    AIE.core(%tile0_2) {
        %subviewIn = AIE.objectfifo.acquire @of_in (Consume, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        %subviewOut = AIE.objectfifo.acquire<Produce>(%objFifo_02_01 : !AIE.objectfifo<memref<1024xi32>>, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        
        %input = AIE.objectfifo.subview.access %subviewIn[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %output = AIE.objectfifo.subview.access %subviewOut[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %offset = arith.constant 16 : index
        %blocksize = arith.constant 16 : index

        func.call @autocorrelate(%input, %output, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()

        AIE.objectfifo.release @of_in (Consume, 1)

        // Append the prior results, block of 16.
        %subviewIn3 = AIE.objectfifo.acquire<Consume>(%objFifo_03_02 : !AIE.objectfifo<memref<1024xi32>>, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        %input3 = AIE.objectfifo.subview.access %subviewIn3[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

        %1 = memref.subview %input3[0][64][1] : memref<1024xi32> to memref<64xi32>
        %2 = memref.subview %output[%blocksize][64][1] : memref<1024xi32> to memref<64xi32, strided<[1], offset: ?>>
        memref.copy %1, %2 : memref<64xi32> to memref<64xi32, strided<[1], offset: ?>>
        
        AIE.objectfifo.release<Consume>(%objFifo_03_02 : !AIE.objectfifo<memref<1024xi32>>, 1)

        AIE.objectfifo.release<Produce>(%objFifo_02_01 : !AIE.objectfifo<memref<1024xi32>>, 1)
        
        AIE.end
    }

    AIE.core(%tile0_3) {
        %subviewIn = AIE.objectfifo.acquire @of_in (Consume, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        %subviewOut = AIE.objectfifo.acquire<Produce>(%objFifo_03_02 : !AIE.objectfifo<memref<1024xi32>>, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        
        %input = AIE.objectfifo.subview.access %subviewIn[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %output = AIE.objectfifo.subview.access %subviewOut[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %offset = arith.constant 32 : index
        %blocksize = arith.constant 16 : index

        func.call @autocorrelate(%input, %output, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()

        AIE.objectfifo.release @of_in (Consume, 1)

        // Append the prior results, block of 16.
        %subviewIn4 = AIE.objectfifo.acquire<Consume>(%objFifo_04_03 : !AIE.objectfifo<memref<1024xi32>>, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        %input4 = AIE.objectfifo.subview.access %subviewIn4[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

        %1 = memref.subview %input4[0][64][1] : memref<1024xi32> to memref<64xi32>
        %2 = memref.subview %output[%blocksize][64][1] : memref<1024xi32> to memref<64xi32, strided<[1], offset: ?>>
        memref.copy %1, %2 : memref<64xi32> to memref<64xi32, strided<[1], offset: ?>>
        
        AIE.objectfifo.release<Consume>(%objFifo_04_03 : !AIE.objectfifo<memref<1024xi32>>, 1)

        AIE.objectfifo.release<Produce>(%objFifo_03_02 : !AIE.objectfifo<memref<1024xi32>>, 1)
        
        AIE.end
    }

    AIE.core(%tile0_4) {
        %subviewIn = AIE.objectfifo.acquire @of_in (Consume, 1) : !AIE.objectfifosubview<memref<1024xi32>>
        %subviewOut = AIE.objectfifo.acquire<Produce>(%objFifo_04_03 : !AIE.objectfifo<memref<1024xi32>>, 1) : !AIE.objectfifosubview<memref<1024xi32>>

        %input = AIE.objectfifo.subview.access %subviewIn[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %output = AIE.objectfifo.subview.access %subviewOut[0] : !AIE.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %offset = arith.constant 48 : index
        %blocksize = arith.constant 16 : index

        func.call @autocorrelate(%input, %output, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()

        AIE.objectfifo.release<Produce>(%objFifo_04_03 : !AIE.objectfifo<memref<1024xi32>>, 1)
        AIE.objectfifo.release @of_in (Consume, 1)
        
        AIE.end
    }
}
