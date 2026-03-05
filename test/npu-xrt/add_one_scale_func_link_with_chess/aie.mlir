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
// End-to-end test for func-level link_with with multiple .o files (Chess/xbridge).
//
// Two func.func declarations each carry a distinct link_with attribute.
// aie-assign-core-link-files (run inside aiecc) traces both CallOps inside
// the core and produces link_files = ["add_one_kernel.o", "scale_kernel.o"]
// on the CoreOp.  The BCF emitter turns each into an _include _file directive,
// and xbridge links both .o files into the core ELF.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

    // Two func-level link_withs — each refers to a different .o file.
    // aie-assign-core-link-files aggregates both into the core's link_files.
    func.func private @add_one(memref<8xi32>, memref<8xi32>, i32) attributes {link_with = "add_one_kernel.o"}
    func.func private @scale_by_two(memref<8xi32>, memref<8xi32>, i32) attributes {link_with = "scale_kernel.o"}

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

        // Step 1: add_one_kernel.o — out[i] = in[i] + 1
        func.call @add_one(%elem_in, %elem_out, %n) : (memref<8xi32>, memref<8xi32>, i32) -> ()
        // Step 2: scale_kernel.o — out[i] = out[i] * 2 (in-place via two-pointer form)
        func.call @scale_by_two(%elem_out, %elem_out, %n) : (memref<8xi32>, memref<8xi32>, i32) -> ()

        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%in : memref<64xi32>, %out : memref<64xi32>) {
      %c0  = arith.constant 0 : i64
      %c1  = arith.constant 1 : i64
      %c8  = arith.constant 8 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1])  {metadata = @of_in,  id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
