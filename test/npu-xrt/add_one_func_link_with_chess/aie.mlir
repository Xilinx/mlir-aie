//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// End-to-end test for func-level link_with (Chess/xbridge backend).
//
// A func.func declaration carries {link_with = "add_one_kernel.o"}.  The
// aie-assign-core-link-files pass (run inside aiecc) traces the CallOp inside
// the core and populates the core's link_files attribute, which the BCF emitter
// turns into _include _file directives consumed by xbridge.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

    // func-level link_with: the kernel .o is declared here, not on aie.core.
    func.func private @add_one(memref<8xi32>, memref<8xi32>, i32) attributes {link_with = "add_one_kernel.o"}

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %n  = arith.constant 8 : i32

      scf.for %i = %c0 to %c8 step %c1 {
        %sub_in  = aie.objectfifo.acquire @of_in(Consume, 1)  : !aie.objectfifosubview<memref<8xi32>>
        %elem_in = aie.objectfifo.subview.access %sub_in[0]   : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %sub_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem_out = aie.objectfifo.subview.access %sub_out[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>

        func.call @add_one(%elem_in, %elem_out, %n) : (memref<8xi32>, memref<8xi32>, i32) -> ()

        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%in : memref<64xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0  = arith.constant 0 : i64
      %c1  = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1])  {metadata = @of_in,  id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
