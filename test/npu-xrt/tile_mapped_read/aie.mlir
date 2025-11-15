//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @read_processor_bus(memref<8xi32>, i32, i32, i32)
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
  
    aie.objectfifo @objFifo_in0(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @objFifo_in1(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])

    aie.objectfifo @objFifo_out1(%t02, {%t01}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @objFifo_out0(%t01, {%t00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])
  
    // Create 8 locks on tile 0,2 with init values 42 to 49
    // This is the data the core will read over the processor bus
    aie.lock(%t02, 8) {init = 42 : i32, sym_name = "lock8"}
    aie.lock(%t02, 9) {init = 43 : i32, sym_name = "lock9"}
    aie.lock(%t02, 10) {init = 44 : i32, sym_name = "lock10"}
    aie.lock(%t02, 11) {init = 45 : i32, sym_name = "lock11"}
    aie.lock(%t02, 12) {init = 46 : i32, sym_name = "lock12"}
    aie.lock(%t02, 13) {init = 47 : i32, sym_name = "lock13"}
    aie.lock(%t02, 14) {init = 48 : i32, sym_name = "lock14"}
    aie.lock(%t02, 15) {init = 49 : i32, sym_name = "lock15"}

    aie.core(%t02) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %size = arith.constant 8 : i32
      %stride = arith.constant 0x10 : i32
      %addr = arith.constant 0x0001F080 : i32
      scf.for %steps = %c0 to %c8 step %c1 {
        %subview0 = aie.objectfifo.acquire @objFifo_in1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %subview1 = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @read_processor_bus(%elem1, %addr, %size, %stride) : (memref<8xi32>, i32, i32, i32) -> ()
        aie.objectfifo.release @objFifo_in1(Consume, 1)
        aie.objectfifo.release @objFifo_out1(Produce, 1)
      }
      aie.end
    } {link_with = "kernel.o"}

    aie.runtime_sequence(%in : memref<64xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      // Set Core_Processor_Bus register Enable = 1. Without this the core will hang on access to the processor bus
      aiex.npu.maskwrite32 {address = 0x32038 : ui32, row = 2 : i32, column = 0 : i32, value = 0x1 : ui32, mask = 0x1 : ui32}
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_in0, id = 0 : i64 } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_out0, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }

}
