//===- stack_size_indirect_call.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// resolveIndirectCallEdges' conservative function-pointer-table inference
// (see StackSizeAnalysis.h's file comment) is otherwise untested: every other
// stack-size fixture only exercises direct func.call-derived relocations.
// indirect_caller (see stack_size_indirect_call_kernel.cc) calls
// target_fn only through a function pointer loaded from a global, so the
// only way the analysis can fold target_fn's real (~4096-byte) frame into
// indirect_caller's path is by combining "target_fn's address escapes into
// g_dispatch" with "indirect_caller references g_dispatch" into a
// conservative call edge.
//
// stack_size = 2048 sits below indirect_caller's own (trivial) frame plus
// target_fn's real one, so correct inference must warn (and, since 2048 is
// explicit and genuinely insufficient, ultimately fail) naming a large
// number; if the indirect edge were never synthesized, indirect_caller would
// appear to call nothing and the computed requirement would silently be just
// its own tiny frame -- no warning at all.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -fstack-size-section -c %S/stack_size_indirect_call_kernel.cc -o %t.d/stack_size_indirect_call_kernel.o
// RUN: cd %t.d && not %aiecc %s 2>&1 | FileCheck %s

// CHECK: warning: this core's callees need at least {{[0-9][0-9][0-9][0-9]+}} bytes of stack (not counting the core body's own frame), but stack_size is only 2048 bytes
// CHECK: error: stack_size = 2048 is insufficient

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @indirect_caller(memref<512xi8>) attributes {link_with = "stack_size_indirect_call_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @indirect_caller(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 2048 : i32 }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
