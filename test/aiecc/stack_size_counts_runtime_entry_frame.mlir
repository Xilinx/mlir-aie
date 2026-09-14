//===- stack_size_counts_runtime_entry_frame.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's live call chain starts above the core body: `__start` (crt0) calls
// `_main_init` (crt1), which calls the core body. `_main_init`'s frame stays
// live across the whole chain. crt1 belongs to the toolchain, so the compiler
// produces no object that holds it. The linker adds it, and aiecc measures the
// linked core.
//
// An undercount is silent. The stack sits directly below the buffers with no
// clearance, so a core that needs more than it declares overwrites the buffer
// above its stack.
//
// This test reuses the kernel of stack_size_max_not_sum.mlir. The core body
// and its kernels reach exactly 4224 bytes. The check accepts an exact fit, so
// `stack_size = 4224` fails only when the runtime entry frame counts. Take a
// new number from the diagnostic when a peano update moves the frames.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_max_not_sum_kernel.cc -o %t.d/stack_size_max_not_sum_kernel.o
// RUN: cd %t.d && not %aiecc --get=measured_stack_sizes.mlir --output-dir=%t.out %s 2>&1 | FileCheck %s

// The requirement must exceed the 4224 the walk reaches on its own. The core
// must be reported as short, not as an exact fit.
// CHECK: stack_size = 4224 is insufficient: this core needs 42{{[0-9][0-9]}} bytes

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_a(memref<512xi8>) attributes {link_with = "stack_size_max_not_sum_kernel.o"}
    func.func private @entry_b(memref<512xi8>) attributes {link_with = "stack_size_max_not_sum_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
      func.call @entry_a(%e) : (memref<512xi8>) -> ()
      func.call @entry_b(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 4224 : i32 }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
