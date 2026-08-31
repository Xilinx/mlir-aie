//===- stack_size_max_not_sum.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's stack requirement is the maximum over the paths of its roots, since
// one call chain is live at a time. (reserved_data_size sums instead, because
// all static data of the link_files objects coexists.) entry_a and entry_b
// each hold a frame of about 4096 bytes, see the kernel source, and the core
// body calls each one on its own, so the requirement is about one frame, 4
// KiB. stack_size = 5000 sits between one frame and two, so the build stays
// silent under the maximum and fails under a sum. The measured number lands on
// the core as `measured_stack_size`.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_max_not_sum_kernel.cc -o %t.d/stack_size_max_not_sum_kernel.o
// RUN: cd %t.d && %aiecc --get=measured_stack_sizes.mlir --output-dir=%t.out %s 2>&1 | FileCheck --allow-empty %s
// RUN: FileCheck --check-prefix=ATTR %s < %t.out/measured_stack_sizes.mlir

// CHECK-NOT: is insufficient

// ATTR: aie.core(%tile_0_2)
// ATTR: measured_stack_size = 4{{[0-9][0-9][0-9]}} : i32

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
    } { stack_size = 5000 : i32 }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
