//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
  
    aie.objectfifo @objFifo_in0(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @objFifo_in1(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])

    aie.objectfifo @objFifo_out1(%t02, {%t01}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @objFifo_out0(%t01, {%t00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])
  
    aie.core(%t02) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_32 = arith.constant 1 : i32
  
      scf.for %steps = %c0 to %c8 step %c1 {
        %subview0 = aie.objectfifo.acquire @objFifo_in1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %subview1 = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        scf.for %arg3 = %c0 to %c8 step %c1 {
            %0 = memref.load %elem0[%arg3] : memref<8xi32>
            %1 = arith.addi %0, %c1_32 : i32
            memref.store %1, %elem1[%arg3] : memref<8xi32>
        }
        aie.objectfifo.release @objFifo_in1(Consume, 1)
        aie.objectfifo.release @objFifo_out1(Produce, 1)
      }
      aie.end
    }
    aiex.runtime_sequence(%in : memref<64xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_out0, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_in0, id = 0 : i64 } : memref<64xi32>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }
}
