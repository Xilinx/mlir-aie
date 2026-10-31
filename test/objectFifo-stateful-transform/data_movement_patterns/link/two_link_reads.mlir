//===- two_link_reads.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s

// Links on two mem tiles read one broadcast fifo, each into a pool of its own.
// CHECK-LABEL: @two_reads
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_1_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 0)
// CHECK-DAG: aie.buffer(%mem_tile_0_1) {sym_name = "in__cons0__buf0"} : memref<64xi32>
// CHECK-DAG: aie.buffer(%mem_tile_1_1) {sym_name = "in__cons1__buf0"} : memref<64xi32>
module @two_reads {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    aie.objectfifo @in(%s, {%m0, %m1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m0, {%t0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@a] ([] [])
    aie.objectfifo @b(%m1, {%t1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@b] ([] [])
    aie.core(%t0) {
      %x = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %x = aie.objectfifo.acquire @b(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

// -----

// Two links on one mem tile read a fifo that names the tile twice, each
// taking one of its consumer ends.
// CHECK-LABEL: @two_reads_one_tile
// CHECK-DAG: aie.buffer(%mem_tile_0_1) {sym_name = "in__cons0__buf0"} : memref<64xi32>
// CHECK-DAG: aie.buffer(%mem_tile_0_1) {sym_name = "in__cons1__buf0"} : memref<64xi32>
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 1)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 1, %tile_1_2, DMA : 0)
module @two_reads_one_tile {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m0 = aie.tile(0, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    aie.objectfifo @in(%s, {%m0, %m0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m0, {%t0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@a] ([] [])
    aie.objectfifo @b(%m0, {%t1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@b] ([] [])
    aie.core(%t0) {
      %x = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %x = aie.objectfifo.acquire @b(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
