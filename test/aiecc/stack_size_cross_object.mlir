//===- stack_size_cross_object.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Every other stack-size fixture has its whole call graph in one object; this
// one exercises the two-pass cross-object attribution the analysis needs when
// it doesn't (see StackSizeAnalysis.h's file comment): entry_cross (defined in
// stack_size_cross_object_caller.o, the only symbol the core body calls
// directly) calls helper_cross, which is defined only in the *sibling*
// object stack_size_cross_object_callee.o and has the large real frame
// (~4096 bytes). The relocation entry_cross's object carries for that call is
// against an undefined symbol with an unreliable type, so the analysis can
// only recognize it as a call once it has scanned every object in
// link_files for what it defines -- not just the one object being walked.
//
// link_files is set directly on the core (rather than inferred by
// aie-assign-core-link-files from a func.func's link_with) specifically so
// both objects reach this one core without a second, direct func.call to
// helper_cross -- which would make it a root in its own right and defeat the
// point: helper_cross must only be reachable through the cross-object edge.
//
// stack_size = 2048 sits well below entry_cross's own (trivial) frame plus
// helper_cross's real one, so a build that folds the cross-object frame in
// correctly must warn (and, since 2048 is explicit and genuinely
// insufficient, ultimately fail) naming a large number; a broken cross-object
// edge would silently drop helper_cross's contribution and report only
// entry_cross's own tiny frame instead, producing no warning at all.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -fstack-size-section -c %S/stack_size_cross_object_caller.cc -o %t.d/stack_size_cross_object_caller.o
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -fstack-size-section -c %S/stack_size_cross_object_callee.cc -o %t.d/stack_size_cross_object_callee.o
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
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
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
