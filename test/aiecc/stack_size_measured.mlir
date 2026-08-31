//===- stack_size_measured.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc measures the stack requirement of the core from the frame of its
// compiled body and a call-graph walk of the symbols it calls directly, through
// the `.stack_sizes` data of their link_files objects. stack_size holds 1 byte,
// so the requirement exceeds it for any compiler version. The test matches the
// error that names both numbers, and matches the silence under
// --no-auto-stack-size.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_measured_kernel.cc -o %t.d/stack_size_measured_kernel.o
// RUN: cd %t.d && not %aiecc %s 2>&1 | FileCheck %s
// RUN: cd %t.d && %aiecc --no-auto-stack-size %s 2>&1 | FileCheck --check-prefix=NOAUTO --allow-empty %s

// CHECK: error: stack_size = 1 is insufficient: this core needs {{[0-9]+}} bytes; increase stack_size to {{[0-9]+}} (Worker(stack_size=...) in IRON), or pass --no-auto-stack-size to skip this check

// NOAUTO-NOT: is insufficient

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_fn(memref<512xi8>) attributes {link_with = "stack_size_measured_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
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
