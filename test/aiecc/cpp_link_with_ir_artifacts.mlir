//===- cpp_link_with_ir_artifacts.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// LLVM IR link artifacts (.ll/.bc) -- what ExternalFunction(inline=True)
// emits -- are merged into the core's LLVM module by aiecc via llvm-link and
// inlined, not object-linked.  So the ldscript and BCF emitters must leave
// them out of INPUT() / `_include _file`; emitting them there would ask the
// linker to consume LLVM IR as an object, and (once llvm-link has already
// merged them) would define the same symbols twice.
//
// aie-assign-core-link-files itself does not filter: link_files carries every
// artifact, and routing is the emitters' job.  Both halves are checked here.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | FileCheck %s --check-prefix=OPT
// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 | FileCheck %s --check-prefix=LDSCRIPT
// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 | FileCheck %s --check-prefix=BCF

// The pass records all three, unfiltered.
// OPT: link_files = ["kernel_obj.o", "kernel_ir.ll", "kernel_ir.bc"]

// Only the object is handed to the linker.
// LDSCRIPT-NOT: kernel_ir.ll
// LDSCRIPT-NOT: kernel_ir.bc
// LDSCRIPT: INPUT(kernel_obj.o)
// LDSCRIPT-NOT: kernel_ir.ll
// LDSCRIPT-NOT: kernel_ir.bc

// BCF-NOT: kernel_ir.ll
// BCF-NOT: kernel_ir.bc
// BCF: _include _file kernel_obj.o
// BCF-NOT: kernel_ir.ll
// BCF-NOT: kernel_ir.bc

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // Object-linked kernel (default ExternalFunction path).
    func.func private @obj_kernel(memref<16xi32>, memref<16xi32>) attributes {link_with = "kernel_obj.o"}
    // Inlined kernels: textual and bitcode LLVM IR.
    func.func private @ll_kernel(memref<16xi32>, memref<16xi32>) attributes {link_with = "kernel_ir.ll"}
    func.func private @bc_kernel(memref<16xi32>, memref<16xi32>) attributes {link_with = "kernel_ir.bc"}

    %core_0_2 = aie.core(%tile_0_2) {
      %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      func.call @obj_kernel(%elem_in, %elem_out) : (memref<16xi32>, memref<16xi32>) -> ()
      func.call @ll_kernel(%elem_in, %elem_out) : (memref<16xi32>, memref<16xi32>) -> ()
      func.call @bc_kernel(%elem_in, %elem_out) : (memref<16xi32>, memref<16xi32>) -> ()

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
