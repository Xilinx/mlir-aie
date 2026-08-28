//===- default_stack_size.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// --default-stack-size stands in for the target's built-in default for any
// core with stack_size absent, treating it as if written explicitly.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_max_not_sum_kernel.cc -o %t.d/stack_size_max_not_sum_kernel.o

// RUN: cd %t.d && not %aiecc %s 2>&1 | FileCheck --check-prefix=ABSENT %s
// ABSENT: stack_size is absent (this core's buffers were placed assuming the device default of 1024 bytes)

// RUN: cd %t.d && not %aiecc --default-stack-size=2048 %s 2>&1 | FileCheck --check-prefix=TOOSMALL %s
// TOOSMALL: stack_size = 2048 is insufficient (this core's buffers were placed assuming 2048 bytes)
// TOOSMALL-NOT: stack_size is absent

// RUN: cd %t.d && %aiecc --default-stack-size=8192 %s 2>&1 | FileCheck --check-prefix=SUFFICIENT --allow-empty %s
// SUFFICIENT-NOT: is insufficient
// SUFFICIENT-NOT: stack_size is absent
// SUFFICIENT-NOT: this core's callees need at least

// tile_0_3's own explicit stack_size must survive untouched.
// RUN: cd %t.d && %aiecc --default-stack-size=8192 --get=default_stack_size.mlir --output-dir=%t.pop %s
// RUN: FileCheck --check-prefix=POPULATED %s < %t.pop/default_stack_size.mlir
// POPULATED: aie.core(%tile_0_2) {
// POPULATED: } {stack_size = 8192 : i32}
// POPULATED: aie.core(%tile_0_3) {
// POPULATED: } {stack_size = 222 : i32}

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_a(memref<512xi8>) attributes {link_with = "stack_size_max_not_sum_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
      func.call @entry_a(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    %core_0_3 = aie.core(%tile_0_3) {
      aie.end
    } { stack_size = 222 : i32 }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
