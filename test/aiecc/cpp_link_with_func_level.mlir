//===- cpp_link_with_func_level.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Canonical new style: link_with is on func.func, not on aie.core.
// Verify that AIEAssignCoreLinkFiles populates link_files on the core and
// that the ldscript/BCF emitters produce the correct entries.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | FileCheck %s --check-prefix=OPT
// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 | FileCheck %s --check-prefix=LDSCRIPT
// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 | FileCheck %s --check-prefix=BCF

// OPT: link_files = ["f.o"]

// LDSCRIPT: INPUT(f.o)

// BCF: _include _file f.o

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    func.func private @f(memref<16xi32>, memref<16xi32>) attributes {link_with = "f.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      func.call @f(%elem_in, %elem_out) : (memref<16xi32>, memref<16xi32>) -> ()

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
