//===- stack_size_indirect_call.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Exercises the function-pointer inference of resolveIndirectCallEdges (see
// StackSizeAnalysis.h). indirect_caller, in
// stack_size_indirect_call_kernel.cc, reaches target_fn through a function
// pointer loaded from a global. The analysis combines the escape of target_fn
// into g_dispatch with the reference from indirect_caller to g_dispatch, and
// adds the 4096-byte frame of target_fn to the path of indirect_caller.
//
// stack_size = 2048 sits below the full requirement, so the build warns and
// then fails with a large number. Under a missed edge the analysis reports the
// small frame of indirect_caller alone.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_indirect_call_kernel.cc -o %t.d/stack_size_indirect_call_kernel.o
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
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
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
