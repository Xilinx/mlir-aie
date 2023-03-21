//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: aiecc.py -j4 --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib%/ %extraAieCcFlags% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @host_multirate {
    %tile34 = AIE.tile(3, 4)
    %tile70 = AIE.tile(7, 0)

    %ext_buf70_in  = AIE.external_buffer {sym_name = "ddr_test_buffer_in"}: memref<256xi32> 
    %ext_buf70_out1 = AIE.external_buffer {sym_name = "ddr_test_buffer_out1"}: memref<64xi32> 
    %ext_buf70_out2 = AIE.external_buffer {sym_name = "ddr_test_buffer_out2"}: memref<64xi32> 

    %objFifo_in = AIE.objectFifo.createObjectFifo(%tile70, {%tile34}, 1) : !AIE.objectFifo<memref<64xi32>>
    %objFifo_out = AIE.objectFifo.createObjectFifo(%tile34, {%tile70}, 1) : !AIE.objectFifo<memref<64xi32>>

    AIE.objectFifo.registerExternalBuffers(%tile70, %objFifo_in : !AIE.objectFifo<memref<64xi32>>, {%ext_buf70_in}) : (memref<256xi32>)
    AIE.objectFifo.registerExternalBuffers(%tile70, %objFifo_out : !AIE.objectFifo<memref<64xi32>>, {%ext_buf70_out1, %ext_buf70_out2}) : (memref<64xi32>, memref<64xi32>)
 
    %core34 = AIE.core(%tile34) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %height = arith.constant 64 : index
        %init1 = arith.constant 1 : i32

        scf.for %iter = %c0 to %c4 step %c1 { 
            %inputSubview = AIE.objectFifo.acquire<Consume>(%objFifo_in : !AIE.objectFifo<memref<64xi32>>, 1) : !AIE.objectFifoSubview<memref<64xi32>>
            %outputSubview = AIE.objectFifo.acquire<Produce>(%objFifo_out : !AIE.objectFifo<memref<64xi32>>, 1) : !AIE.objectFifoSubview<memref<64xi32>>
            
            %input = AIE.objectFifo.subview.access %inputSubview[0] : !AIE.objectFifoSubview<memref<64xi32>> -> memref<64xi32>
            %output = AIE.objectFifo.subview.access %outputSubview[0] : !AIE.objectFifoSubview<memref<64xi32>> -> memref<64xi32>

            scf.for %indexInHeight = %c0 to %height step %c1 { 
                %d1 = memref.load %input[%indexInHeight] : memref<64xi32>
                memref.store %d1, %output[%indexInHeight] : memref<64xi32> 
            }
            
            AIE.objectFifo.release<Consume>(%objFifo_in : !AIE.objectFifo<memref<64xi32>>, 1)
            AIE.objectFifo.release<Produce>(%objFifo_out : !AIE.objectFifo<memref<64xi32>>, 1)
        }
        AIE.end
    } 
}
