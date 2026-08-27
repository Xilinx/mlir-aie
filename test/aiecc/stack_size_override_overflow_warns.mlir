//===- stack_size_override_overflow_warns.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// stack_size_override is looked up directly as a subtree total in
// maxPathFrom, so an absurdly large (but non-negative, so
// stack_size_override's own validation lets it through) override is a
// simple, direct way to drive the computed requirement above what an i32
// attribute holds -- the same risk a malformed .stack_sizes entry from a
// hostile object file poses, without needing to hand-craft one. Silently
// truncating that to i32 (as populateReservedDataSize's own INT32_MAX guard
// already exists to prevent for reserved_data_size) would wrap to a small or
// negative value and undercount; this must instead warn and leave
// stack_size unvalidated for the core, never stamp a truncated number.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -c %S/stack_size_unmeasurable_kernel.cc -o %t.d/stack_size_unmeasurable_kernel.o
// RUN: cd %t.d && %aiecc %s 2>&1 | FileCheck %s

// CHECK: warning: stack requirement computed as 5000000000 bytes, which does not fit in the attribute's i32; stack_size is not being validated for this core

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @touch_scratch(memref<512xi8>) attributes {link_with = "stack_size_unmeasurable_kernel.o", stack_size_override = 5000000000 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @touch_scratch(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
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
