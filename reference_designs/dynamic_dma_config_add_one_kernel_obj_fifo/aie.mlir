//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: hsa, chess

// RUN: xchesscc_wrapper aie -I %aietools/include -c %S/kernel.cc -o ./kernel.o
// RUN: aiecc.py --link_against_hsa --xchesscc %S/aie.mlir -I%HSA_DIR%/include -L%HSA_DIR%/lib -lhsa-runtime64 -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o %T/test.elf
// RUN: %run_on_board %T/test.elf

module {
  aie.device(xcvc1902) {
    %t70 = aie.tile(6, 0)
    %t71 = aie.tile(6, 1)
    %t72 = aie.tile(6, 2)

    aie.objectfifo @objFifo_in0(%t70, {%t72}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @objFifo_out0(%t72, {%t70}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

    // Need to define the data movement before any other func
    func.func @sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0]) {id = 0 : i64, metadata = @objFifo_out0} : memref<16xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0]) {id = 1 : i64, metadata = @objFifo_in0} : memref<16xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }

    func.func private @func(%AL: memref<8xi32>, %BL: memref<8xi32>) -> ()

    aie.core(%t72) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c1_32 = arith.constant 1 : i32
      scf.for %steps = %c0 to %c2 step %c1 {
        %subview0 = aie.objectfifo.acquire @objFifo_in0(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %subview1 = aie.objectfifo.acquire @objFifo_out0(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @func(%elem0, %elem1) : (memref<8xi32>, memref<8xi32>) -> ()
        aie.objectfifo.release @objFifo_in0(Consume, 1)
        aie.objectfifo.release @objFifo_out0(Produce, 1)

    }

    aie.end

    } { link_with = "kernel.o" }
    
  }
}

