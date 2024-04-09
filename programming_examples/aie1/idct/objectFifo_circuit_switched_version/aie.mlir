//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license && jackl
// RUN: xchesscc -p me -P %aietools/data/versal_prod/lib -c %S/../kernel.cc %S/../dequant.cc %S/../pass.cc
// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @idct {
  %t74 = aie.tile(7, 4)
  %t75 = aie.tile(7, 5)

  %t73 = aie.tile(7, 3)
  %t72 = aie.tile(7, 2)
  %t71 = aie.tile(7, 1)
  %t70 = aie.tile(7, 0)

  aie.objectfifo @of_in (%t70, {%t73}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
  aie.objectfifo @of_dequant_horizontal (%t73, {%t74}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
  aie.objectfifo @of_horizontal_vertical (%t74, {%t75}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
  aie.objectfifo @of_out (%t75, {%t70}, 2 : i32) : !aie.objectfifo<memref<64xi16>>

  // DDR buffer
  %buffer_in  = aie.external_buffer { sym_name = "buffer_in" }  : memref<512xi16>
  %buffer_out = aie.external_buffer { sym_name = "buffer_out" }  : memref<512xi16>

  aie.objectfifo.register_external_buffers @of_in (%t70, {%buffer_in}) : (memref<512xi16>)
  aie.objectfifo.register_external_buffers @of_out (%t70, {%buffer_out}) : (memref<512xi16>)

  func.func private @dequant_8x8(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func.func private @idct_8x8_mmult_h(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func.func private @idct_8x8_mmult_v(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func.func private @pass(%A: memref<64xi16>, %B: memref<64xi16>) -> ()

  %c13 = aie.core(%t73) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index
    
    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    scf.for %iv = %lb to %ub step %step {
      %inputSubview = aie.objectfifo.acquire @of_in (Consume, 1) : !aie.objectfifosubview<memref<64xi16>>
      %input = aie.objectfifo.subview.access %inputSubview[0] : !aie.objectfifosubview<memref<64xi16>> -> memref<64xi16>
      %outputSubview = aie.objectfifo.acquire @of_dequant_horizontal (Produce, 1) : !aie.objectfifosubview<memref<64xi16>>
      %output = aie.objectfifo.subview.access %outputSubview[0] : !aie.objectfifosubview<memref<64xi16>> -> memref<64xi16>

      func.call @dequant_8x8(%input, %output) : (memref<64xi16>, memref<64xi16>) -> ()
      
      aie.objectfifo.release @of_in (Consume, 1)
      aie.objectfifo.release @of_dequant_horizontal (Produce, 1)
    }

    aie.end
  } { link_with="dequant.o" }

  %c74 = aie.core(%t74) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index
    
    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    scf.for %iv = %lb to %ub step %step {
      %inputSubview = aie.objectfifo.acquire @of_dequant_horizontal (Consume, 1) : !aie.objectfifosubview<memref<64xi16>>
      %input = aie.objectfifo.subview.access %inputSubview[0] : !aie.objectfifosubview<memref<64xi16>> -> memref<64xi16>
      %outputSubview = aie.objectfifo.acquire @of_horizontal_vertical (Produce, 1) : !aie.objectfifosubview<memref<64xi16>>
      %output = aie.objectfifo.subview.access %outputSubview[0] : !aie.objectfifosubview<memref<64xi16>> -> memref<64xi16>

      func.call @idct_8x8_mmult_h(%input, %output) : (memref<64xi16>, memref<64xi16>) -> ()
      
      aie.objectfifo.release @of_dequant_horizontal (Consume, 1)
      aie.objectfifo.release @of_horizontal_vertical (Produce, 1)
    }

    aie.end
  } { link_with="idct_horizontal.o" }

  %c75 = aie.core(%t75) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index
    
    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    scf.for %iv = %lb to %ub step %step {
      %inputSubview = aie.objectfifo.acquire @of_horizontal_vertical <Consume>(Consume, 1) : !aie.objectfifosubview<memref<64xi16>>
      %input = aie.objectfifo.subview.access %inputSubview[0] : !aie.objectfifosubview<memref<64xi16>> -> memref<64xi16>
      %outputSubview = aie.objectfifo.acquire @of_out (Produce, 1) : !aie.objectfifosubview<memref<64xi16>>
      %output = aie.objectfifo.subview.access %outputSubview[0] : !aie.objectfifosubview<memref<64xi16>> -> memref<64xi16>

      func.call @idct_8x8_mmult_v(%input, %output) : (memref<64xi16>, memref<64xi16>) -> ()
      
      aie.objectfifo.release @of_horizontal_vertical <Consume>(Consume, 1)
      aie.objectfifo.release @of_out (Produce, 1)
    }

    aie.end
  } { link_with="idct_vertical.o" }
}
