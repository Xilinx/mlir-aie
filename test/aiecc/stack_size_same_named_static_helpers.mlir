//===- stack_size_same_named_static_helpers.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A `static` symbol is only visible in its defining object, so two objects
// in one core's link_files can define an unrelated `helper` under the same
// name. stack_size_same_named_static_a_kernel.cc's `helper` is plain and
// non-recursive, called by entry_a; b_kernel.cc's `helper` is self-recursive
// but only reachable from unused_entry_b, which this core never calls.
//
// If the two `helper`s alias to one call-graph node, entry_a's path inherits
// b's unrelated self-loop and the build fails with a false "recursion
// detected". stack_size = 8192 comfortably covers entry_a's real
// requirement, so correct attribution must build clean.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_same_named_static_a_kernel.cc -o %t.d/stack_size_same_named_static_a_kernel.o
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_same_named_static_b_kernel.cc -o %t.d/stack_size_same_named_static_b_kernel.o
// RUN: cd %t.d && %aiecc %s 2>&1 | FileCheck --allow-empty %s

// CHECK-NOT: recursion detected
// CHECK-NOT: error:

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_a(memref<512xi8>)

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @entry_a(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } {
      stack_size = 8192 : i32,
      link_files = ["stack_size_same_named_static_a_kernel.o", "stack_size_same_named_static_b_kernel.o"]
    }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
