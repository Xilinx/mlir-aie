//===- test_sa_core_sections_fallback.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles="placer=sa_placer sa-seed=3" %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %t | FileCheck %s --check-prefix=ADDR

// The SA charges a core's own sections as assign-buffer-addresses reserves
// them: data_size, then measured_data_size, then nothing. The producer (0, 3)
// declares neither, so with its 1 KB stack it is the emptier end of the 62 KB
// shared-memory fifo and keeps it; the consumer (0, 2) holds a 2 KB buffer and
// a 512 B data region. The same holds with the cores physical or pinned
// logical.
// CHECK-NOT: aie.objectfifo.allocate

// ADDR-LABEL: module @physical
// ADDR-DAG: aie.buffer(%{{.*}}tile_0_3) {{.*}}sym_name = "f__buf0"
// ADDR-DAG: aie.buffer(%{{.*}}tile_0_2) {{.*}}sym_name = "small"
// ADDR-LABEL: module @pinned
// ADDR-DAG: aie.buffer(%{{.*}}tile_0_3) {{.*}}sym_name = "f__buf0"
// ADDR-DAG: aie.buffer(%{{.*}}tile_0_2) {{.*}}sym_name = "small"

module @physical {
  aie.device(npu2_1col) {
    %p = aie.tile(0, 3)
    %c = aie.tile(0, 2)
    %lc = aie.logical_tile<CoreTile>(0, 5)
    %b = aie.buffer(%c) {sym_name = "small"} : memref<512xi32>
    aie.objectfifo @f(%p, {%c}, 2 : i32) : !aie.objectfifo<memref<7936xi32>>
    aie.core(%p) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<7936xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<7936xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    } {data_size = 512 : i32}
    aie.core(%lc) { aie.end }
  }
}

// -----

module @pinned {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(0, 3)
    %c = aie.logical_tile<CoreTile>(0, 2)
    %lc = aie.logical_tile<CoreTile>(0, 5)
    %b = aie.buffer(%c) {sym_name = "small"} : memref<512xi32>
    aie.objectfifo @f(%p, {%c}, 2 : i32) : !aie.objectfifo<memref<7936xi32>>
    aie.core(%p) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<7936xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<7936xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    } {data_size = 512 : i32}
    aie.core(%lc) { aie.end }
  }
}

// -----

// A physical core whose 63 KB buffer and 1 KB stack fill its memory exactly
// fits, as addressing finds; before, the 3 KB estimate made the SA warn.
// ADDR-LABEL: module @exact
// ADDR: aie.buffer(%{{.*}}tile_1_3) {{.*}}sym_name = "big"
module @exact {
  aie.device(npu2) {
    %p = aie.tile(1, 3)
    %b = aie.buffer(%p) {sym_name = "big"} : memref<64512xi8>
    %l1 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%p) { aie.end }
    aie.core(%l1) { aie.end }
  }
}
