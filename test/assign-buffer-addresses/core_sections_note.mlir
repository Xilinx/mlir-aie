//===- core_sections_note.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-place-tiles --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s

// test_sa_physical_core_sections.mlir's design: %pc's tile has no room for
// "big" beside the 8192 bytes its core keeps for its own sections. The note
// saying so sits on that core, so it names the same core whichever of the
// two aie.core ops comes first.
// CHECK: error: 'aie.buffer' op could not be placed: buffer "big"

module @physical_core_sections {
  aie.device(npu2_1col) {
    %pc = aie.tile(0, 3)
    %lc = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%pc) {sym_name = "big"} : memref<1024xi32>
    aie.objectfifo @f(%pc, {%lc}, 2 : i32) : !aie.objectfifo<memref<6656xi32>>
    // CHECK: :[[@LINE+1]]:5: note: this core keeps its own sections in 8192 bytes (device @main)
    aie.core(%pc) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<6656xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    } {data_size = 8192 : i32}
    aie.core(%lc) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<6656xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
  }
}

// -----

module @physical_core_sections_swapped {
  aie.device(npu2_1col) {
    %pc = aie.tile(0, 3)
    %lc = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%pc) {sym_name = "big"} : memref<1024xi32>
    aie.objectfifo @f(%pc, {%lc}, 2 : i32) : !aie.objectfifo<memref<6656xi32>>
    aie.core(%lc) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<6656xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
    // CHECK: error: 'aie.buffer' op could not be placed: buffer "big"
    // CHECK: :[[@LINE+1]]:5: note: this core keeps its own sections in 8192 bytes (device @main)
    aie.core(%pc) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<6656xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    } {data_size = 8192 : i32}
  }
}
