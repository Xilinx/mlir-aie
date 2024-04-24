//===- aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: Novembre 10th 2022
// 
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: xchesscc -p me -P %aietools/data/versal_prod/lib -c %S/../kernel.cc
// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

aie.device(xcvc1902) {
  %t60 = aie.tile(6, 0)
  %t63 = aie.tile(6, 3)
  %t64 = aie.tile(6, 4)

  %t70 = aie.tile(7, 0)
  %t73 = aie.tile(7, 3)
  %t74 = aie.tile(7, 4)

  %t100 = aie.tile(10, 0)

  aie.objectfifo @of_LHS0 (%t60, {%t63, %t73}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
  aie.objectfifo @of_LHS1 (%t60, {%t64, %t74}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
  aie.objectfifo @of_RHS0 (%t70, {%t63}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
  aie.objectfifo @of_RHS1 (%t70, {%t64}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
  aie.objectfifo @of_RHS2 (%t100, {%t73}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
  aie.objectfifo @of_RHS3 (%t100, {%t74}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>

  aie.objectfifo @of_out0 (%t64, {%t60}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
  aie.objectfifo @of_out1 (%t74, {%t60}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>

  aie.objectfifo @of_acc0 (%t63, {%t64}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
  aie.objectfifo @of_acc1 (%t73, {%t74}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>

  %buffer0 = aie.external_buffer {sym_name = "LHS_tile0"} : memref<1024 x i32>     //LHS_tile0
  %buffer1 = aie.external_buffer {sym_name = "LHS_tile1"} : memref<1024 x i32>     //LHS_tile1
  %buffer2 = aie.external_buffer {sym_name = "RHS_tile0"} : memref<1024 x i32>     //RHS_tile0
  %buffer3 = aie.external_buffer {sym_name = "RHS_tile1"} : memref<1024 x i32>     //RHS_tile1
  %buffer4 = aie.external_buffer {sym_name = "RHS_tile2"} : memref<1024 x i32>     //RHS_tile2
  %buffer5 = aie.external_buffer {sym_name = "RHS_tile3"} : memref<1024 x i32>     //RHS_tile3
  %buffer6 = aie.external_buffer {sym_name = "Out_tile0"} : memref<1024 x i32>     //Out_tile0
  %buffer7 = aie.external_buffer {sym_name = "Out_tile1"} : memref<1024 x i32>     //Out_tile1

  aie.objectfifo.register_external_buffers @of_LHS0 (%t60, {%buffer0}) : (memref<1024xi32>)
  aie.objectfifo.register_external_buffers @of_LHS1 (%t60, {%buffer1}) : (memref<1024xi32>)
  aie.objectfifo.register_external_buffers @of_out0 (%t60, {%buffer6}) : (memref<1024xi32>)
  aie.objectfifo.register_external_buffers @of_out1 (%t60, {%buffer7}) : (memref<1024xi32>)

  aie.objectfifo.register_external_buffers @of_RHS0 (%t70, {%buffer2}) : (memref<1024xi32>)
  aie.objectfifo.register_external_buffers @of_RHS1 (%t70, {%buffer3}) : (memref<1024xi32>)

  aie.objectfifo.register_external_buffers @of_RHS2 (%t100, {%buffer4}) : (memref<1024xi32>)
  aie.objectfifo.register_external_buffers @of_RHS3 (%t100, {%buffer5}) : (memref<1024xi32>)

  %buf63 = aie.buffer(%t63) {sym_name = "buf63"} : memref<1024xi32>  //Accumulator0
  %buf73 = aie.buffer(%t73) {sym_name = "buf73"} : memref<1024xi32>  //Accumulator1

  func.func private @extern_kernel(%A: memref<1024xi32>, %B: memref<1024xi32>, %acc: memref<1024xi32>, %C: memref<1024xi32>) -> ()

  %core63 = aie.core(%t63) {
    %LHS0Subview = aie.objectfifo.acquire @of_LHS0 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %LHS0 = aie.objectfifo.subview.access %LHS0Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %RHS0Subview = aie.objectfifo.acquire @of_RHS0 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %RHS0 = aie.objectfifo.subview.access %RHS0Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %ACC0Subview = aie.objectfifo.acquire @of_acc0 (Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %ACC0 = aie.objectfifo.subview.access %ACC0Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    func.call @extern_kernel(%LHS0, %RHS0, %buf63, %ACC0) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    
    aie.objectfifo.release @of_LHS0 (Consume, 1)
    aie.objectfifo.release @of_RHS0 (Consume, 1)
    aie.objectfifo.release @of_acc0 (Produce, 1)

    aie.end
  } { link_with="kernel.o" }

  %core64 = aie.core(%t64) {
    %LHS1Subview = aie.objectfifo.acquire @of_LHS1 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %LHS1 = aie.objectfifo.subview.access %LHS1Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %RHS1Subview = aie.objectfifo.acquire @of_RHS1 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %RHS1 = aie.objectfifo.subview.access %RHS1Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %ACC0Subview = aie.objectfifo.acquire @of_acc0 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %ACC0 = aie.objectfifo.subview.access %ACC0Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %Out0Subview = aie.objectfifo.acquire @of_out0 (Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %Out0 = aie.objectfifo.subview.access %Out0Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    func.call @extern_kernel(%LHS1, %RHS1, %ACC0, %Out0) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    
    aie.objectfifo.release @of_LHS1 (Consume, 1)
    aie.objectfifo.release @of_RHS1 (Consume, 1)
    aie.objectfifo.release @of_acc0 (Consume, 1)
    aie.objectfifo.release @of_out0 (Produce, 1)

    aie.end
  } { link_with="kernel.o" }

  %core73 = aie.core(%t73) {
    %LHS0Subview = aie.objectfifo.acquire @of_LHS0 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %LHS0 = aie.objectfifo.subview.access %LHS0Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %RHS2Subview = aie.objectfifo.acquire @of_RHS2 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %RHS2 = aie.objectfifo.subview.access %RHS2Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %ACC1Subview = aie.objectfifo.acquire @of_acc1 (Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %ACC1 = aie.objectfifo.subview.access %ACC1Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    func.call @extern_kernel(%LHS0, %RHS2, %buf73, %ACC1) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    
    aie.objectfifo.release @of_LHS0 (Consume, 1)
    aie.objectfifo.release @of_RHS2 (Consume, 1)
    aie.objectfifo.release @of_acc1 (Produce, 1)

    aie.end
  } { link_with="kernel.o" }

  %core74 = aie.core(%t74) {
    %LHS1Subview = aie.objectfifo.acquire @of_LHS1 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %LHS1 = aie.objectfifo.subview.access %LHS1Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %RHS3Subview = aie.objectfifo.acquire @of_RHS3 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %RHS3 = aie.objectfifo.subview.access %RHS3Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %ACC1Subview = aie.objectfifo.acquire @of_acc1 (Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %ACC1 = aie.objectfifo.subview.access %ACC1Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    %Out1Subview = aie.objectfifo.acquire @of_out1 (Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
    %Out1 = aie.objectfifo.subview.access %Out1Subview[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>

    func.call @extern_kernel(%LHS1, %RHS3, %ACC1, %Out1) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    
    aie.objectfifo.release @of_LHS1 (Consume, 1)
    aie.objectfifo.release @of_RHS3 (Consume, 1)
    aie.objectfifo.release @of_acc1 (Consume, 1)
    aie.objectfifo.release @of_out1 (Produce, 1)

    aie.end
  } { link_with="kernel.o" }
}
