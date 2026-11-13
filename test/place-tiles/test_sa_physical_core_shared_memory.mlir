//===- test_sa_physical_core_shared_memory.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=3" %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %t | FileCheck %s --check-prefix=ADDR

// A shared-memory fifo between the physical core (0, 3) and a logical core
// keeps its objects on whichever tile the SA charged them to, which is the
// one without the 40 KB buffer. The SA writes the aie.objectfifo.allocate
// for a physical end as for a pinned logical one. Before, it skipped fifos
// with a physical end, so allocation kept the objects on the producer and
// addressing failed ("f__buf1 needs 16384 bytes and this tile has no room
// left") although the SA reported no overrun.

// CHECK-LABEL: module @physical_producer
// CHECK: aie.objectfifo.allocate @f(%tile_0_2)
// ADDR-LABEL: module @physical_producer
// ADDR-DAG: aie.buffer(%tile_0_2) {{.*}}sym_name = "f__buf0"
// ADDR-DAG: aie.buffer(%tile_0_2) {{.*}}sym_name = "f__buf1"
module @physical_producer {
  aie.device(npu2_1col) {
    %pc = aie.tile(0, 3)
    %lc = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%pc) {sym_name = "big"} : memref<10240xi32>
    aie.objectfifo @f(%pc, {%lc}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.core(%pc) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<4096xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
    aie.core(%lc) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<4096xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @physical_consumer
// CHECK: aie.objectfifo.allocate @f(%tile_0_3)
// ADDR-LABEL: module @physical_consumer
// ADDR-DAG: aie.buffer(%tile_0_3) {{.*}}sym_name = "f__buf0"
// ADDR-DAG: aie.buffer(%tile_0_3) {{.*}}sym_name = "f__buf1"
module @physical_consumer {
  aie.device(npu2_1col) {
    %pc = aie.tile(0, 3)
    %lc = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%lc) {sym_name = "big"} : memref<10240xi32>
    aie.objectfifo @f(%lc, {%pc}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.core(%pc) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<4096xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
    aie.core(%lc) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<4096xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
  }
}
