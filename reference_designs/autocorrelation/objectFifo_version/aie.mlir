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

// XFAIL: *

module @autocorrelation {
    %tile0_1 = aie.tile(2, 1)
    %tile0_2 = aie.tile(2, 2)
    %tile0_3 = aie.tile(2, 3)
    %tile0_4 = aie.tile(2, 4)
    %tile0_5 = aie.tile(2, 5)
    %tile0_6 = aie.tile(2, 6)
    %tile0_7 = aie.tile(2, 7)
    %tile0_8 = aie.tile(2, 8)

    %tile7_0 = aie.tile(7, 0)

    %inputExt = aie.external_buffer {sym_name = "input"} : memref<1024 x i32>
    %outputExt = aie.external_buffer {sym_name = "output"} : memref<1024 x i32>

    aie.objectfifo @of_in (%tile7_0, {%tile0_1, %tile0_2, %tile0_3, %tile0_4}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @of_out (%tile0_1, {%tile7_0}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>

    aie.objectfifo.register_external_buffers(%tile7_0, %objFifoIn : !aie.objectfifo<memref<1024xi32>>, {%inputExt}) : (memref<1024xi32>)
    aie.objectfifo.register_external_buffers(%tile7_0, %objFifoOut : !aie.objectfifo<memref<1024xi32>>, {%outputExt}) : (memref<1024xi32>)

    aie.objectfifo @of_04_03 (%tile0_4, {%tile0_3}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @of_03_02 (%tile0_3, {%tile0_2}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @of_02_01 (%tile0_2, {%tile0_1}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>

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

    aie.core(%tile0_1) {
        %subviewIn = aie.objectfifo.acquire @of_in (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %subviewOut = aie.objectfifo.acquire @of_out (Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
        
        %input = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %output = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %offset = arith.constant 0 : index
        %blocksize = arith.constant 16 : index

        func.call @autocorrelate(%input, %output, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()

        aie.objectfifo.release @of_in (Consume, 1)

        // Append the prior results, block of 16.
        %subviewIn2 = aie.objectfifo.acquire @of_in (%objFifo_02_01 : !aie.objectfifo<memref<1024xi32>>, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %input2 = aie.objectfifo.subview.access %subviewIn2[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

        %1 = memref.subview %input2[0][64][1] : memref<1024xi32> to memref<64xi32>
        %2 = memref.subview %output[%blocksize][64][1] : memref<1024xi32> to memref<64xi32, strided<[1], offset: ?>>
        memref.copy %1, %2 : memref<64xi32> to memref<64xi32, strided<[1], offset: ?>>
        
        aie.objectfifo.release<Consume>(%objFifo_02_01 : !aie.objectfifo<memref<1024xi32>>, 1)

        aie.objectfifo.release @of_out (Produce, 1)
        
        aie.end
    }

    aie.core(%tile0_2) {
        %subviewIn = aie.objectfifo.acquire @of_in (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %subviewOut = aie.objectfifo.acquire<Produce>(%objFifo_02_01 : !aie.objectfifo<memref<1024xi32>>, 1) : !aie.objectfifosubview<memref<1024xi32>>
        
        %input = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %output = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %offset = arith.constant 16 : index
        %blocksize = arith.constant 16 : index

        func.call @autocorrelate(%input, %output, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()

        aie.objectfifo.release @of_in (Consume, 1)

        // Append the prior results, block of 16.
        %subviewIn3 = aie.objectfifo.acquire<Consume>(%objFifo_03_02 : !aie.objectfifo<memref<1024xi32>>, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %input3 = aie.objectfifo.subview.access %subviewIn3[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

        %1 = memref.subview %input3[0][64][1] : memref<1024xi32> to memref<64xi32>
        %2 = memref.subview %output[%blocksize][64][1] : memref<1024xi32> to memref<64xi32, strided<[1], offset: ?>>
        memref.copy %1, %2 : memref<64xi32> to memref<64xi32, strided<[1], offset: ?>>
        
        aie.objectfifo.release<Consume>(%objFifo_03_02 : !aie.objectfifo<memref<1024xi32>>, 1)

        aie.objectfifo.release<Produce>(%objFifo_02_01 : !aie.objectfifo<memref<1024xi32>>, 1)
        
        aie.end
    }

    aie.core(%tile0_3) {
        %subviewIn = aie.objectfifo.acquire @of_in (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %subviewOut = aie.objectfifo.acquire<Produce>(%objFifo_03_02 : !aie.objectfifo<memref<1024xi32>>, 1) : !aie.objectfifosubview<memref<1024xi32>>
        
        %input = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %output = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %offset = arith.constant 32 : index
        %blocksize = arith.constant 16 : index

        func.call @autocorrelate(%input, %output, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()

        aie.objectfifo.release @of_in (Consume, 1)

        // Append the prior results, block of 16.
        %subviewIn4 = aie.objectfifo.acquire<Consume>(%objFifo_04_03 : !aie.objectfifo<memref<1024xi32>>, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %input4 = aie.objectfifo.subview.access %subviewIn4[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

        %1 = memref.subview %input4[0][64][1] : memref<1024xi32> to memref<64xi32>
        %2 = memref.subview %output[%blocksize][64][1] : memref<1024xi32> to memref<64xi32, strided<[1], offset: ?>>
        memref.copy %1, %2 : memref<64xi32> to memref<64xi32, strided<[1], offset: ?>>
        
        aie.objectfifo.release<Consume>(%objFifo_04_03 : !aie.objectfifo<memref<1024xi32>>, 1)

        aie.objectfifo.release<Produce>(%objFifo_03_02 : !aie.objectfifo<memref<1024xi32>>, 1)
        
        aie.end
    }

    aie.core(%tile0_4) {
        %subviewIn = aie.objectfifo.acquire @of_in (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %subviewOut = aie.objectfifo.acquire<Produce>(%objFifo_04_03 : !aie.objectfifo<memref<1024xi32>>, 1) : !aie.objectfifosubview<memref<1024xi32>>

        %input = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %output = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %offset = arith.constant 48 : index
        %blocksize = arith.constant 16 : index

        func.call @autocorrelate(%input, %output, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()

        aie.objectfifo.release<Produce>(%objFifo_04_03 : !aie.objectfifo<memref<1024xi32>>, 1)
        aie.objectfifo.release @of_in (Consume, 1)
        
        aie.end
    }
}
