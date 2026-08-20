//===- stack_size_measured.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The core's stack requirement is auto-measured from a call-graph walk of
// the symbols it directly calls (entry_fn -> helper_fn here, both in the
// same object) through their link_files objects' `.stack_sizes` metadata.
// stack_size below is deliberately far too small (1 byte) so the computed
// requirement always exceeds it, without pinning an exact byte count that
// would be fragile across compiler versions -- only that a warning fires
// naming both numbers, and that --no-auto-stack-size suppresses it entirely.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -fstack-size-section -c %S/stack_size_measured_kernel.cc -o %t.d/stack_size_measured_kernel.o
// RUN: cd %t.d && %aiecc %s 2>&1 | FileCheck %s
// RUN: cd %t.d && %aiecc --no-auto-stack-size %s 2>&1 | FileCheck --check-prefix=NOAUTO --allow-empty %s

// CHECK: warning: this core's callees need at least {{[0-9]+}} bytes of stack (not counting the core body's own frame), but stack_size is only 1 bytes

// NOAUTO-NOT: this core's callees need at least

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_fn(memref<512xi8>) attributes {link_with = "stack_size_measured_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @entry_fn(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 1 : i32 }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
