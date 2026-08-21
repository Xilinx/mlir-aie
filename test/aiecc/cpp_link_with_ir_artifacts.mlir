//===- cpp_link_with_ir_artifacts.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// How a link artifact reaches the core is decided by metadata, not by the file
// name.  `link_with_mode = "merge"` -- what ExternalFunction(inline=True)
// emits -- routes the artifact into `link_merge_files`, which aiecc llvm-links
// into the core's LLVM module and inlines; the ldscript and BCF emitters must
// leave those out of INPUT() / `_include _file`, or (once llvm-link has merged
// them) the same symbols would be defined twice.  Everything else lands in
// `link_files` and is an ordinary final-link input whatever its suffix -- a
// .bc included, since lld accepts bitcode as an LTO input.
//
// This file owns the emitters' half of the contract.  The pass's routing --
// which artifact lands in which list -- is pinned by
// cpp_link_with_mode_merge.mlir and is deliberately not re-asserted here.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 | FileCheck %s --check-prefix=LDSCRIPT
// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 | FileCheck %s --check-prefix=BCF

// Every link_files entry is handed to the linker verbatim -- including the
// .bc, which lld takes as an LTO input -- and the merge-mode artifact appears
// nowhere.
// LDSCRIPT-NOT:  kernel_merge.ll
// LDSCRIPT:      INPUT(kernel_obj.o)
// LDSCRIPT-NEXT: INPUT(kernel_ir.bc)
// LDSCRIPT-NOT:  kernel_merge.ll

// BCF-NOT:  kernel_merge.ll
// BCF:      _include _file kernel_obj.o
// BCF-NEXT: _include _file kernel_ir.bc
// BCF-NOT:  kernel_merge.ll

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // Ordinary link inputs (default ExternalFunction path): an object, and a
    // bitcode module that is object-linked because it asks for no other mode.
    func.func private @obj_kernel(memref<16xi32>, memref<16xi32>) attributes {link_with = "kernel_obj.o"}
    func.func private @bc_kernel(memref<16xi32>, memref<16xi32>) attributes {link_with = "kernel_ir.bc"}
    // Inlined kernel: ExternalFunction(inline=True) marks the mode explicitly.
    func.func private @merge_kernel(memref<16xi32>, memref<16xi32>) attributes {link_with = "kernel_merge.ll", link_with_mode = "merge"}

    %core_0_2 = aie.core(%tile_0_2) {
      %elem_in = aie.objectfifo.acquire @of_in (Consume, 1) : memref<16xi32>

      %elem_out = aie.objectfifo.acquire @of_out (Produce, 1) : memref<16xi32>

      func.call @obj_kernel(%elem_in, %elem_out) : (memref<16xi32>, memref<16xi32>) -> ()
      func.call @bc_kernel(%elem_in, %elem_out) : (memref<16xi32>, memref<16xi32>) -> ()
      func.call @merge_kernel(%elem_in, %elem_out) : (memref<16xi32>, memref<16xi32>) -> ()

      aie.objectfifo.release @of_in (Consume, 1)
      aie.objectfifo.release @of_out (Produce, 1)
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
