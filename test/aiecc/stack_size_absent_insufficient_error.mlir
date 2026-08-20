//===- stack_size_absent_insufficient_error.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// stack_size is absent (the common case -- most designs never set it), and
// entry_a's real frame (~4160 bytes, see stack_size_max_not_sum_kernel.cc)
// exceeds the device default (0x400 = 1024 bytes) that this core's buffers
// were placed against. Today this silently corrupts memory at runtime;
// instead, once the core's own compiled frame is known (only possible after
// this build has already run to completion), the build must fail with the
// exact value to declare -- never silently pick one itself, the same
// "compiler measures and reports, user declares and rebuilds" rule as every
// other check in this analysis.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -fstack-size-section -c %S/stack_size_max_not_sum_kernel.cc -o %t.d/stack_size_max_not_sum_kernel.o
// RUN: cd %t.d && not %aiecc %s 2>&1 | FileCheck %s

// CHECK: error: stack_size is absent (this core's buffers were placed assuming the device default of 1024 bytes), but this core's real requirement is {{[0-9]+}} bytes; set stack_size = {{[0-9]+}} explicitly on this aie.core and rebuild, or pass --no-auto-stack-size to skip this check

// --no-auto-stack-size skips this check entirely, same as the earlier
// warning it complements -- the build must complete despite the same
// underlying insufficiency.
// RUN: rm -rf %t.noauto.d && mkdir -p %t.noauto.d
// RUN: cp %t.d/stack_size_max_not_sum_kernel.o %t.noauto.d/
// RUN: cd %t.noauto.d && %aiecc --no-auto-stack-size %s 2>&1 | FileCheck --check-prefix=NOAUTO --allow-empty %s

// NOAUTO-NOT: stack_size is absent

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_a(memref<512xi8>) attributes {link_with = "stack_size_max_not_sum_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @entry_a(%e) : (memref<512xi8>) -> ()
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
