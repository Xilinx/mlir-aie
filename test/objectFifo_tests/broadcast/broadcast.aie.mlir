//===- broadcast.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: September 15th 2022
// 
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

// This test 

module @broadcast {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)
    %tile14 = AIE.tile(1, 4)
    %tile33 = AIE.tile(3, 3)

    %buff_out_12 = AIE.buffer(%tile12) { sym_name = "out12" } :  memref<2x16xi32>
    %lock_out_12 = AIE.lock(%tile12, 0)
    %buff_out_14 = AIE.buffer(%tile14) { sym_name = "out14" } :  memref<2x16xi32>
    %lock_out_14 = AIE.lock(%tile14, 0)
    %buff_out_33 = AIE.buffer(%tile33) { sym_name = "out33" } :  memref<2x16xi32>
    %lock_out_33 = AIE.lock(%tile33, 0)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile13, {%tile12, %tile14, %tile33}, 4) : !AIE.objectFifo<memref<16xi32>>

    func.func @storeLineScalar(%lineIn : memref<16xi32>, %row : index, %bufferOut : memref<2x16xi32>) -> () {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %lineWidth = arith.constant 16 : index

        scf.for %indexInLine = %c0 to %lineWidth step %c1 {
            %value = memref.load %lineIn[%indexInLine] : memref<16xi32>
            memref.store %value, %bufferOut[%row,%indexInLine] : memref<2x16xi32>
        }
        return
    }

    %core13 = AIE.core(%tile13) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        
        %subview0 = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0) : (memref<16xi32>) -> ()
        AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)

        %subview1 = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem1) : (memref<16xi32>) -> ()
        AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        
        AIE.end
    }

    %core12 = AIE.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index

        AIE.useLock(%lock_out_12, "Acquire", 0)

        %subview0 = AIE.objectFifo.acquire<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @storeLineScalar(%elem0, %c0, %buff_out_12) : (memref<16xi32>, index, memref<2x16xi32>) -> ()
        AIE.objectFifo.release<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)

        %subview1 = AIE.objectFifo.acquire<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @storeLineScalar(%elem0, %c1, %buff_out_12) : (memref<16xi32>, index, memref<2x16xi32>) -> ()
        AIE.objectFifo.release<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)

        AIE.useLock(%lock_out_12, "Release", 1)
        
        AIE.end
    }

    %core14 = AIE.core(%tile14) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index

        AIE.useLock(%lock_out_14, "Acquire", 0)
        
        %subview = AIE.objectFifo.acquire<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem1 = AIE.objectFifo.subview.access %subview[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @storeLineScalar(%elem0, %c0, %buff_out_14) : (memref<16xi32>, index, memref<2x16xi32>) -> ()
        func.call @storeLineScalar(%elem1, %c1, %buff_out_14) : (memref<16xi32>, index, memref<2x16xi32>) -> ()
        AIE.objectFifo.release<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 2)
            
        AIE.useLock(%lock_out_14, "Release", 1)
        
        AIE.end
    }

    %core33 = AIE.core(%tile33) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index

        AIE.useLock(%lock_out_33, "Acquire", 0)

        %subview = AIE.objectFifo.acquire<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem1 = AIE.objectFifo.subview.access %subview[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @storeLineScalar(%elem0, %c0, %buff_out_33) : (memref<16xi32>, index, memref<2x16xi32>) -> ()
        func.call @storeLineScalar(%elem1, %c1, %buff_out_33) : (memref<16xi32>, index, memref<2x16xi32>) -> ()
        AIE.objectFifo.release<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)

        AIE.useLock(%lock_out_33, "Release", 1)
        
        AIE.end
    }
}