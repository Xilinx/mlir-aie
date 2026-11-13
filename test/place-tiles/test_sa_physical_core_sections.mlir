//===- test_sa_physical_core_sections.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=3" %s | FileCheck %s
// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %t | FileCheck %s --check-prefix=ADDR

// The physical core (0, 3) holds a 4 KB buffer, its 1 KB stack and an 8 KB
// data region. The SA counts its stack and data region as it does a logical
// core's, so the 52 KB shared-memory fifo goes to the logical consumer and
// the SA writes its aie.objectfifo.allocate. Before, it saw only the 4 KB
// buffer, kept the fifo on (0, 3), and addressing failed ("buffer "big"
// needs 4096 bytes and this tile has no room left") after a clean report.
// CHECK: aie.objectfifo.allocate @f(%tile_0_{{[24]}})
// ADDR-DAG: aie.buffer(%tile_0_{{[24]}}) {{.*}}sym_name = "f__buf0"
// ADDR-DAG: aie.buffer(%tile_0_{{[24]}}) {{.*}}sym_name = "f__buf1"
// ADDR-DAG: aie.buffer(%tile_0_3) {{.*}}sym_name = "big"

module @physical_core_sections {
  aie.device(npu2_1col) {
    %pc = aie.tile(0, 3)
    %lc = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%pc) {sym_name = "big"} : memref<1024xi32>
    aie.objectfifo @f(%pc, {%lc}, 2 : i32) : !aie.objectfifo<memref<6656xi32>>
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
