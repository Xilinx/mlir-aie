//===- bank_region_overflow.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A table larger than its bank has nowhere to go. Placement now measures what
// a core's objects pin to each bank, so this is caught while reserving the room
// rather than later by the linker, and the diagnostic can name the bank, the
// demand and the capacity.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: sed 's/tbl_ab\[256\]/tbl_ab[8192]/' %S/bank_section_placed_kernel.cc > %t.d/oversized.cc
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I%S/../../third_party/aie_api/include -ffunction-sections -fdata-sections -c %t.d/oversized.cc -o %t.d/bank_section_placed_kernel.o
// RUN: cd %t.d && not aiecc --get-core-elfs %s 2>&1 | FileCheck %s

// CHECK: error: {{.*}}static data pinned to bank 1 requires {{[0-9]+}} bytes,
// CHECK-SAME: which cannot fit in bank 1 ({{[0-9]+}} bytes total)
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    func.func private @classify(memref<64xi8>) attributes {link_with = "bank_section_placed_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<64xi8>
      func.call @classify(%e) : (memref<64xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 1024 : i32 }

    aie.runtime_sequence(%out : memref<64xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
