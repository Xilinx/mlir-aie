//===- stack_size_explicit_override_wins.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The same recursive kernel that fails outright in
// stack_size_recursion_error.mlir must build cleanly once its root symbol
// carries stack_size_override: the override cuts the subtree beneath it, so
// the analysis never trips over the self-recursive `recurse` inside.
//
// The core's own stack_size (8192) comfortably covers the 4096-byte override
// plus its own frame, so the later, more complete sufficiency check (see
// stack_size_absent_insufficient_error.mlir) has nothing to say either.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_recursive_kernel.cc -o %t.d/stack_size_recursive_kernel.o
// RUN: cd %t.d && %aiecc %s 2>&1 | FileCheck --allow-empty %s

// CHECK-NOT: cannot determine this core's stack requirement
// CHECK-NOT: recursion detected

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @recursive_touch(memref<512xi8>) attributes {link_with = "stack_size_recursive_kernel.o", stack_size_override = 4096 : i32}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
      func.call @recursive_touch(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 8192 : i32 }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
