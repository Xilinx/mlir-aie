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
// RUN: aiecc.py -j4 --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib%/ %extraAieCcFlags% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o host_loop.exe
// RUN: %run_on_board ./host_loop.exe

module @host_loop {
    %tile34 = AIE.tile(3, 4)
    %tile70 = AIE.tile(7, 0)

    %ext_buf70_in  = AIE.external_buffer {sym_name = "ddr_test_buffer_in"}: memref<256xi32> 
    %ext_buf70_out = AIE.external_buffer {sym_name = "ddr_test_buffer_out"}: memref<256xi32> 

    %objFifo_in = AIE.objectFifo.createObjectFifo(%tile70, {%tile34}, 1) : !AIE.objectFifo<memref<256xi32>>
    %objFifo_out = AIE.objectFifo.createObjectFifo(%tile34, {%tile70}, 1) : !AIE.objectFifo<memref<256xi32>>

    AIE.objectFifo.registerExternalBuffers(%tile70, %objFifo_in : !AIE.objectFifo<memref<256xi32>>, {%ext_buf70_in}) : (memref<256xi32>)
    AIE.objectFifo.registerExternalBuffers(%tile70, %objFifo_out : !AIE.objectFifo<memref<256xi32>>, {%ext_buf70_out}) : (memref<256xi32>)
 
    %core34 = AIE.core(%tile34) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c25 = arith.constant 25 : index
        %height = arith.constant 256 : index
        
        scf.for %iter = %c0 to %c25 step %c1 { 
            %inputSubview = AIE.objectFifo.acquire<Consume>(%objFifo_in : !AIE.objectFifo<memref<256xi32>>, 1) : !AIE.objectFifoSubview<memref<256xi32>>
            %outputSubview = AIE.objectFifo.acquire<Produce>(%objFifo_out : !AIE.objectFifo<memref<256xi32>>, 1) : !AIE.objectFifoSubview<memref<256xi32>>
            
            %input = AIE.objectFifo.subview.access %inputSubview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
            %output = AIE.objectFifo.subview.access %outputSubview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

            scf.for %indexInHeight = %c0 to %height step %c1 { 
                %d1 = memref.load %input[%indexInHeight] : memref<256xi32>
                memref.store %d1, %output[%indexInHeight] : memref<256xi32> 
            }
            
            AIE.objectFifo.release<Consume>(%objFifo_in : !AIE.objectFifo<memref<256xi32>>, 1)
            AIE.objectFifo.release<Produce>(%objFifo_out : !AIE.objectFifo<memref<256xi32>>, 1)
        }
            
        AIE.end
    } 
}
