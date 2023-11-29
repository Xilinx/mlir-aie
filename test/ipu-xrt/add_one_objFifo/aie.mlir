//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  AIE.device(ipu) {
    %t00 = AIE.tile(0, 0)
    %t01 = AIE.tile(0, 1)
    %t02 = AIE.tile(0, 2)
  
    AIE.objectfifo @objFifo_in0(%t00, {%t01}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>
    AIE.objectfifo @objFifo_in1(%t01, {%t02}, 2 : i32) : !AIE.objectfifo<memref<8xi32>>
    AIE.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ()
    AIE.objectfifo @objFifo_out0(%t01, {%t00}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>
    AIE.objectfifo @objFifo_out1(%t02, {%t01}, 2 : i32) : !AIE.objectfifo<memref<8xi32>>
    AIE.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ()
  
    AIE.core(%t02) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_32 = arith.constant 1 : i32
  
      scf.for %steps = %c0 to %c8 step %c1 {
        %subview0 = AIE.objectfifo.acquire @objFifo_in1(Consume, 1) : !AIE.objectfifosubview<memref<8xi32>>
        %elem0 = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %subview1 = AIE.objectfifo.acquire @objFifo_out1(Produce, 1) : !AIE.objectfifosubview<memref<8xi32>>
        %elem1 = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        scf.for %arg3 = %c0 to %c8 step %c1 {
            %0 = memref.load %elem0[%arg3] : memref<8xi32>
            %1 = arith.addi %0, %c1_32 : i32
            memref.store %1, %elem1[%arg3] : memref<8xi32>
        }
        AIE.objectfifo.release @objFifo_in1(Consume, 1)
        AIE.objectfifo.release @objFifo_out1(Produce, 1)
      }
      AIE.end
    }
    func.func @sequence(%in : memref<64xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c64 = arith.constant 64 : i32
      AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0]) { metadata = @objFifo_out0, id = 1 : i32 } : (i32, i32, memref<64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0]) { metadata = @objFifo_in0, id = 0 : i32 } : (i32, i32, memref<64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      AIEX.ipu.sync { column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
      return
    }
  }
}
