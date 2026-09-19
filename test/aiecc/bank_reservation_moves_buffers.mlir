//===- bank_reservation_moves_buffers.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Buffers must give way to the banks a kernel's static data needs.
//
// This is the design that motivated measuring objects before placing buffers.
// Two bank-sized buffers and two small tables, one pinned to bank 0 and one to
// bank 1. A tile is 64 kB in four 16 kB banks, so each buffer fills a bank
// exactly and only two placements exist that leave the tables anywhere to go:
// the buffers must take banks 2 and 3.
//
// Placing buffers first cannot find that. The allocator has not seen the
// kernel's object -- it has not been compiled yet -- so it puts the buffers in
// the first banks that fit, the tables are left a zero-length region, and the
// link fails with "section '.aie.bank1' will not fit in region 'bank1'". The
// only recourse was to pin the buffers by hand with mem_bank.
//
// Nothing here pins anything. The test is that it links at all.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -ffunction-sections -fdata-sections -c %S/bank_reservation_moves_buffers_kernel.cc -o %t.d/bank_reservation_moves_buffers_kernel.o
// RUN: cd %t.d && aiecc --get-core-elfs %s
// RUN: llvm-readelf -s %t.d/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s

// Local memory starts at 0x70000 and a bank is 0x4000. Match the bank rather
// than an exact address: where inside a bank a reservation lands is the
// allocator's business.
// CHECK-DAG: 0007{{[0-3][0-9a-f]+}} {{.*}} table_a
// CHECK-DAG: 0007{{[4-7][0-9a-f]+}} {{.*}} table_b

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // One bank each. Unpinned: finding room for the tables is the allocator's
    // job, not the design's.
    %buf_in = aie.buffer(%tile_0_2) {sym_name = "buf_in"} : memref<16384xi8>
    %buf_out = aie.buffer(%tile_0_2) {sym_name = "buf_out"} : memref<16384xi8>

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    func.func private @classify(memref<64xi8>) attributes {link_with = "bank_reservation_moves_buffers_kernel.o"}

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
