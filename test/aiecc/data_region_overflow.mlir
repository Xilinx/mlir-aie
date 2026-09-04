//===- data_region_overflow.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core whose own sections do not fit alongside its buffers.
//
// The linker reports the shortfall against the `data` region. It cannot say
// where that region came from, so aiecc names the core, the script that
// declares the region, and what bounds it.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -c %S/data_region_overflow_kernel.cc -o %t.d/data_region_overflow_kernel.o
// RUN: cd %t.d && not aiecc --get-core-elfs %s 2>&1 | FileCheck %s

// CHECK: will not fit in region 'data'
// CHECK: aiecc: core {{.*}}_core_0_2: its .data/.rodata/.bss exceed the `data` region of ldScripts_{{.*}}_core_0_2.ld.script
// CHECK-SAME: largest gap between this core's stack and this tile's buffers
// CHECK-SAME: set reserved_data_size on the core

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Buffers large enough that the remaining space cannot hold the kernel's
    // 40000-byte .bss, in any placement order.
    %big0 = aie.buffer(%tile_0_2) { sym_name = "big0" } : memref<16384xi8>
    %big1 = aie.buffer(%tile_0_2) { sym_name = "big1" } : memref<16384xi8>

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    func.func private @touch(memref<64xi8>) attributes {link_with = "data_region_overflow_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %v0 = memref.load %big0[%c0] : memref<16384xi8>
      memref.store %v0, %big1[%c0] : memref<16384xi8>
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<64xi8>
      func.call @touch(%e) : (memref<64xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%out : memref<64xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
