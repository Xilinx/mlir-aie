//===- stack_size_compiler_rt.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The kernel calls __divsf3 and __modsi3, which libclang_rt.builtins.a carries
// without `.stack_sizes`. Their frames come from the compiler-rt table, so the
// core measures exactly on both targets.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_compiler_rt_kernel.cc -o %t.d/stack_size_compiler_rt_kernel.o
// RUN: cd %t.d && %aiecc --get=input_with_addresses.mlir --get=measured_stack_sizes.mlir --output-dir=%t.out %s 2>&1 | FileCheck --allow-empty %s
// RUN: FileCheck --check-prefix=ATTR %s < %t.out/input_with_addresses.mlir

// RUN: rm -rf %t.aie2.d && mkdir -p %t.aie2.d
// RUN: clang++ --target=aie2-none-unknown-elf -std=c++20 -O2 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_compiler_rt_kernel.cc -o %t.aie2.d/stack_size_compiler_rt_kernel.o
// RUN: sed 's/aie.device(npu2)/aie.device(npu1_1col)/' %s > %t.aie2.d/npu1.mlir
// RUN: cd %t.aie2.d && %aiecc --get=input_with_addresses.mlir --get=measured_stack_sizes.mlir --output-dir=%t.aie2.out npu1.mlir 2>&1 | FileCheck --allow-empty %s
// RUN: FileCheck --check-prefix=ATTR %s < %t.aie2.out/input_with_addresses.mlir

// CHECK-NOT: no stack size information
// ATTR: measured_stack_size = {{[0-9]+}} : i32

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    func.func private @div_mod(memref<16xi32>) attributes {link_with = "stack_size_compiler_rt_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<16xi32>
      func.call @div_mod(%e) : (memref<16xi32>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
