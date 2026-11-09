//===- test_sa_core_section_reservations.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=2" %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: not aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %t 2>&1 >/dev/null | FileCheck %s --check-prefix=ADDR

// The SA charges each core what assign-buffer-addresses reserves for its own
// sections (CoreOp::getSectionReservations), measured bank sizes and a
// prebaked ELF's data ranges included. Before, it charged the data region
// only.

// The producer (0, 3) pins 3 KB of static data to bank 0, so the consumer is
// the emptier end of the 61 KB shared-memory fifo and the SA moves it there.
// Before, the SA kept it on (0, 3) and addressing failed ("f__buf1 needs
// 31232 bytes and this tile has no room left").
// CHECK-LABEL: module @bank_sizes
// CHECK: aie.objectfifo.allocate @f(%tile_0_2)
// ADDR-NOT: error
module @bank_sizes {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(0, 3)
    %c = aie.logical_tile<CoreTile>(0, 2)
    %b = aie.buffer(%c) {sym_name = "small"} : memref<512xi32>
    aie.objectfifo @f(%p, {%c}, 2 : i32) : !aie.objectfifo<memref<7808xi32>>
    aie.core(%p) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<7808xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    } {measured_bank_sizes = array<i32: 3072, 0, 0, 0>, measured_bank_alignments = array<i32: 16, 16, 16, 16>}
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<7808xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
  }
}

// -----

// A prebaked core's 2 KB data range and its 62 KB buffer do not fit beside
// its stack, and the SA says so as addressing does. Before, it was silent.
// WARN-NOT: over capacity
// WARN: [[@LINE+4]]:3: warning: SA placer's resource estimate is over capacity
// ADDR: error: 'aie.buffer' op could not be placed: buffer "big" needs 63488 bytes
// ADDR-NOT: error
module @prebaked_ranges {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%p) {sym_name = "big"} : memref<15872xi32>
    aie.core(%p) { aie.end } {elf_file = "prebaked.elf", measured_data_ranges = array<i32: 1024, 2048>}
  }
}
