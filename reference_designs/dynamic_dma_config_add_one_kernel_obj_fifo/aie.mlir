//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  aie.device(xcvc1902) {
    %t70 = aie.tile(6, 0)
    %t71 = aie.tile(6, 1)
    %t72 = aie.tile(6, 2)

    aie.objectfifo @objFifo_in0(%t70, {%t72}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @objFifo_out0(%t72, {%t70}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

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
