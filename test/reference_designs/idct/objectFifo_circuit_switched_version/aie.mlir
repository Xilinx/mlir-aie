//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license && jackl
// RUN: xchesscc -p me -P ${CARDANO}/data/cervino/lib -c %S/kernel.cc
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @idct {
  %t74 = AIE.tile(7, 4)
  %t75 = AIE.tile(7, 5)

  %t73 = AIE.tile(7, 3)
  %t72 = AIE.tile(7, 2)
  %t71 = AIE.tile(7, 1)
  %t70 = AIE.tile(7, 0)

  %of_t70_t73 = AIE.objectFifo.createObjectFifo(%t70, {%t73}, 2) : !AIE.objectFifo<memref<64xi16>>
  %of_t73_t74 = AIE.objectFifo.createObjectFifo(%t73, {%t74}, 2) : !AIE.objectFifo<memref<64xi16>>
  %of_t74_t75 = AIE.objectFifo.createObjectFifo(%t74, {%t75}, 2) : !AIE.objectFifo<memref<64xi16>>
  %of_t75_t70 = AIE.objectFifo.createObjectFifo(%t75, {%t70}, 2) : !AIE.objectFifo<memref<64xi16>>

  // DDR buffer
  %buffer_in  = AIE.external_buffer : memref<512xi16>
  %buffer_out = AIE.external_buffer : memref<512xi16>

  AIE.objectFifo.registerExternalBuffers(%t70, %of_t70_t73 : !AIE.objectFifo<memref<64xi16>>, {%buffer_in}) : (memref<512xi16>)
  AIE.objectFifo.registerExternalBuffers(%t70, %of_t75_t70 : !AIE.objectFifo<memref<64xi16>>, {%buffer_out}) : (memref<512xi16>)

  func.func private @dequant_8x8(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func.func private @idct_8x8_mmult_h(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func.func private @idct_8x8_mmult_v(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func.func private @pass(%A: memref<64xi16>, %B: memref<64xi16>) -> ()

  %c13 = AIE.core(%t73) { 
    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index
    
    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    scf.for %iv = %lb to %ub step %step {
      %inputSubview = AIE.objectFifo.acquire<Consume>(%of_t70_t73 : !AIE.objectFifo<memref<64xi16>>, 1) : !AIE.objectFifoSubview<memref<64xi16>>
      %input = AIE.objectFifo.subview.access %inputSubview[0] : !AIE.objectFifoSubview<memref<64xi16>> -> memref<64xi16>
      %outputSubview = AIE.objectFifo.acquire<Produce>(%of_t73_t74 : !AIE.objectFifo<memref<64xi16>>, 1) : !AIE.objectFifoSubview<memref<64xi16>>
      %output = AIE.objectFifo.subview.access %outputSubview[0] : !AIE.objectFifoSubview<memref<64xi16>> -> memref<64xi16>

      func.call @dequant_8x8(%input, %output) : (memref<64xi16>, memref<64xi16>) -> ()
      
      AIE.objectFifo.release<Consume>(%of_t70_t73 : !AIE.objectFifo<memref<64xi16>>, 1)
      AIE.objectFifo.release<Produce>(%of_t73_t74 : !AIE.objectFifo<memref<64xi16>>, 1)
    }

    AIE.end
  } { link_with="dequant.o" }

  %c74 = AIE.core(%t74) { 
    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index
    
    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    scf.for %iv = %lb to %ub step %step {
      %inputSubview = AIE.objectFifo.acquire<Consume>(%of_t73_t74 : !AIE.objectFifo<memref<64xi16>>, 1) : !AIE.objectFifoSubview<memref<64xi16>>
      %input = AIE.objectFifo.subview.access %inputSubview[0] : !AIE.objectFifoSubview<memref<64xi16>> -> memref<64xi16>
      %outputSubview = AIE.objectFifo.acquire<Produce>(%of_t74_t75 : !AIE.objectFifo<memref<64xi16>>, 1) : !AIE.objectFifoSubview<memref<64xi16>>
      %output = AIE.objectFifo.subview.access %outputSubview[0] : !AIE.objectFifoSubview<memref<64xi16>> -> memref<64xi16>

      func.call @idct_8x8_mmult_h(%input, %output) : (memref<64xi16>, memref<64xi16>) -> ()
      
      AIE.objectFifo.release<Consume>(%of_t73_t74 : !AIE.objectFifo<memref<64xi16>>, 1)
      AIE.objectFifo.release<Produce>(%of_t74_t75 : !AIE.objectFifo<memref<64xi16>>, 1)  
    }

    AIE.end
  } { link_with="idct_horizontal.o" }

  %c75 = AIE.core(%t75) { 
    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index
    
    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    scf.for %iv = %lb to %ub step %step {
      %inputSubview = AIE.objectFifo.acquire<Consume>(%of_t74_t75 : !AIE.objectFifo<memref<64xi16>>, 1) : !AIE.objectFifoSubview<memref<64xi16>>
      %input = AIE.objectFifo.subview.access %inputSubview[0] : !AIE.objectFifoSubview<memref<64xi16>> -> memref<64xi16>
      %outputSubview = AIE.objectFifo.acquire<Produce>(%of_t75_t70 : !AIE.objectFifo<memref<64xi16>>, 1) : !AIE.objectFifoSubview<memref<64xi16>>
      %output = AIE.objectFifo.subview.access %outputSubview[0] : !AIE.objectFifoSubview<memref<64xi16>> -> memref<64xi16>

      func.call @idct_8x8_mmult_v(%input, %output) : (memref<64xi16>, memref<64xi16>) -> ()
      
      AIE.objectFifo.release<Consume>(%of_t74_t75 : !AIE.objectFifo<memref<64xi16>>, 1)
      AIE.objectFifo.release<Produce>(%of_t75_t70 : !AIE.objectFifo<memref<64xi16>>, 1)   
    }

    AIE.end
  } { link_with="idct_vertical.o" }
}
