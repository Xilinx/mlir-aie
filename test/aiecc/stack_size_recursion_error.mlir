//===- stack_size_recursion_error.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Recursion is unbounded, so aiecc fails the run unless the root symbol of the
// recursive kernel carries a stack_size_override. The diagnostic names the
// cycle.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_recursive_kernel.cc -o %t.d/stack_size_recursive_kernel.o
// RUN: cd %t.d && not %aiecc %s 2>&1 | FileCheck %s

// CHECK: error: cannot determine this core's stack requirement: recursion detected: __start -> _main_init -> core_0_2 -> recursive_touch -> recurse -> recurse
// CHECK-SAME: set stack_size_override

// --no-measure-stack-size skips the check, including this failure.
// RUN: cd %t.d && %aiecc --no-measure-stack-size %s 2>&1 | FileCheck --check-prefix=NOAUTO --allow-empty %s

// NOAUTO-NOT: cannot determine this core's stack requirement
// NOAUTO-NOT: recursion detected

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @recursive_touch(memref<512xi8>) attributes {link_with = "stack_size_recursive_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
      func.call @recursive_touch(%e) : (memref<512xi8>) -> ()
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
