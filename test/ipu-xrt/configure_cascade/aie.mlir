//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  aie.device(ipu) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)

    aie.configure_cascade(%t02, West, East)
    aie.configure_cascade(%t12, West, East)
  
    aie.objectfifo @objFifo_in0(%t00, {%t01}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @objFifo_in1(%t01, {%t02}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ()

    aie.objectfifo @objFifo_out1(%t12, {%t01}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @objFifo_out0(%t01, {%t00}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ()

    func.func private @extern_kernel1() -> ()
    func.func private @extern_kernel2(%b: memref<64xi32>) -> ()

    %lock13_1 = aie.lock(%t02, 1) { sym_name = "lock_13_1" }
  
    %core02 = aie.core(%t02) {
      %subview0 = aie.objectfifo.acquire @objFifo_in1(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
      aie.use_lock(%lock13_1, "Acquire", 0)

      func.call @extern_kernel1() : () -> ()

      aie.use_lock(%lock13_1, "Release", 1)
      aie.objectfifo.release @objFifo_in1(Consume, 1)

      aie.end
    } { link_with="kernel1.o" }

    %core12 = aie.core(%t12) {
        aie.use_lock(%lock13_1, "Acquire", 1)

        %subview1 = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

        func.call @extern_kernel2(%elem1) : (memref<64xi32>) -> ()

        aie.use_lock(%lock13_1, "Release", 0)
        aie.objectfifo.release @objFifo_out1(Produce, 1)
        aie.end
    } { link_with="kernel2.o" }

    func.func @sequence(%in : memref<64xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.ipu.dma_memcpy_nd (0, 0, %out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0]) { metadata = @objFifo_out0, id = 1 : i64 } : memref<64xi32>
      aiex.ipu.dma_memcpy_nd (0, 0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0]) { metadata = @objFifo_in0, id = 0 : i64 } : memref<64xi32>
      aiex.ipu.sync { column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
      return
    }
  }
}
