//===- stack_size_cross_object.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Exercises the two-pass cross-object attribution (see StackSizeAnalysis.h):
// entry_cross (the core body's only direct call, defined in
// stack_size_cross_object_caller.o) calls helper_cross, defined only in the
// sibling object stack_size_cross_object_callee.o with a large real frame
// (~4096 bytes). link_files is set directly on the core, rather than
// inferred from a func.func's link_with, so both objects reach this core
// without a second direct func.call to helper_cross that would make it a
// root in its own right and bypass the cross-object edge under test.
//
// stack_size = 2048 sits well below the true total, so correct cross-object
// folding must warn and then fail naming a large number; a broken edge would
// silently drop helper_cross's contribution and report no warning at all.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_cross_object_caller.cc -o %t.d/stack_size_cross_object_caller.o
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_cross_object_callee.cc -o %t.d/stack_size_cross_object_callee.o
// RUN: cd %t.d && not %aiecc %s 2>&1 | FileCheck %s

// CHECK: warning: this core's callees need at least {{[0-9][0-9][0-9][0-9]+}} bytes of stack (not counting the core body's own frame), but stack_size is only 2048 bytes
// CHECK: error: stack_size = 2048 is insufficient

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_cross(memref<512xi8>)

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
      func.call @entry_cross(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } {
      stack_size = 2048 : i32,
      link_files = ["stack_size_cross_object_caller.o", "stack_size_cross_object_callee.o"]
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
