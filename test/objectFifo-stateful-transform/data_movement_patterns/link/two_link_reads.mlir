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
// CHECK-LABEL: aie.memtile_dma(%mem_tile_0_1)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.use_lock(%in__cons0__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons0__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.use_lock(%in__cons0__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons0__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<64xi32> offset = 0 len = 64)
// CHECK-LABEL: aie.memtile_dma(%mem_tile_1_1)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.use_lock(%in__cons1__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons1__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.use_lock(%in__cons1__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons1__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<64xi32> offset = 0 len = 64)
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
// CHECK-LABEL: aie.memtile_dma(%mem_tile_0_1)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.use_lock(%in__cons0__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons0__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.dma_start(S2MM, 1
// CHECK:       aie.use_lock(%in__cons1__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons1__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.use_lock(%in__cons0__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons0__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.use_lock(%in__cons1__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons1__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<64xi32> offset = 0 len = 64)
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

// -----

// Two links on one compute tile read a fifo that names the tile twice, each
// taking one of its consumer ends; neither end is read by the tile's core
// (it has none here). The IRON front end's forward() emits this shape onto
// one hub Tile.
// CHECK-LABEL: @two_reads_one_core_tile
// CHECK-DAG: aie.buffer(%tile_0_2) {sym_name = "in__cons0__buf0"} : memref<64xi32>
// CHECK-DAG: aie.buffer(%tile_0_2) {sym_name = "in__cons1__buf0"} : memref<64xi32>
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 1)
// CHECK-DAG: aie.flow(%tile_0_2, DMA : 0, %tile_0_4, DMA : 0)
// CHECK-DAG: aie.flow(%tile_0_2, DMA : 1, %tile_1_4, DMA : 0)
// CHECK-LABEL: aie.mem(%tile_0_2)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.use_lock(%in__cons0__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons0__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.dma_start(S2MM, 1
// CHECK:       aie.use_lock(%in__cons1__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons1__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.use_lock(%in__cons0__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons0__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.use_lock(%in__cons1__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<64xi32> offset = 0 len = 64)
// CHECK:       aie.use_lock(%in__cons1__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<64xi32> offset = 0 len = 64)
module @two_reads_one_core_tile {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %c = aie.tile(0, 2)
    %t0 = aie.tile(0, 4)
    %t1 = aie.tile(1, 4)
    aie.objectfifo @in(%s, {%c, %c}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%c, {%t0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@a] ([] [])
    aie.objectfifo @b(%c, {%t1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
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

// Joins on two mem tiles read the same two broadcast fifos: each producer
// sends once to both mem tiles, and each join fills its own pool, one slot
// per input, from that input's consumer end on its tile.
// CHECK-LABEL: @two_joins
// CHECK-DAG: %[[A0:.*]] = aie.buffer(%mem_tile_0_1) {sym_name = "a__buf0"} : memref<32xi32>
// CHECK-DAG: %[[A1:.*]] = aie.buffer(%mem_tile_0_1) {sym_name = "a__buf1"} : memref<32xi32>
// CHECK-DAG: %[[B0:.*]] = aie.buffer(%mem_tile_1_1) {sym_name = "b__buf0"} : memref<32xi32>
// CHECK-DAG: %[[B1:.*]] = aie.buffer(%mem_tile_1_1) {sym_name = "b__buf1"} : memref<32xi32>
// CHECK-DAG: aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%tile_0_2, DMA : 0, %mem_tile_1_1, DMA : 0)
// CHECK-DAG: aie.flow(%tile_1_2, DMA : 0, %mem_tile_0_1, DMA : 1)
// CHECK-DAG: aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 1)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_3, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_3, DMA : 0)
// CHECK-LABEL: aie.memtile_dma(%mem_tile_0_1)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.use_lock(%a__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%[[A0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[A1]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_start(S2MM, 1
// CHECK:       aie.use_lock(%a__prodlock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%[[A0]] : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_bd(%[[A1]] : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd(%[[A0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[A0]] : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_bd(%[[A1]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[A1]] : memref<32xi32> offset = 16 len = 16)
// CHECK-LABEL: aie.memtile_dma(%mem_tile_1_1)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.use_lock(%b__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%[[B0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[B1]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_start(S2MM, 1
// CHECK:       aie.use_lock(%b__prodlock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%[[B0]] : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_bd(%[[B1]] : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd(%[[B0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[B0]] : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_bd(%[[B1]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[B1]] : memref<32xi32> offset = 16 len = 16)
module @two_joins {
  aie.device(npu2) {
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %p0 = aie.tile(0, 2)
    %p1 = aie.tile(1, 2)
    %t0 = aie.tile(0, 3)
    %t1 = aie.tile(1, 3)
    aie.objectfifo @x(%p0, {%m0, %m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%p1, {%m0, %m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m0, {%t0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@x, @y] -> [@a] ([0, 16] [])
    aie.objectfifo @b(%m1, {%t1}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@x, @y] -> [@b] ([0, 16] [])
    aie.core(%p0) {
      %x = aie.objectfifo.acquire @x(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @x(Produce, 1)
      aie.end
    }
    aie.core(%p1) {
      %x = aie.objectfifo.acquire @y(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @y(Produce, 1)
      aie.end
    }
    aie.core(%t0) {
      %x = aie.objectfifo.acquire @a(Consume, 1) : memref<32xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %x = aie.objectfifo.acquire @b(Consume, 1) : memref<32xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

// -----

// Distributes on two mem tiles split the same broadcast fifo, each filling
// its own pool from the fifo's consumer end on its tile.
// CHECK-LABEL: @two_distributes
// CHECK-DAG: %[[IN0:.*]] = aie.buffer(%mem_tile_0_1) {sym_name = "in__cons0__buf0"} : memref<32xi32>
// CHECK-DAG: %[[IN1:.*]] = aie.buffer(%mem_tile_1_1) {sym_name = "in__cons1__buf0"} : memref<32xi32>
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_1_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_3, DMA : 0)
// CHECK-LABEL: aie.memtile_dma(%mem_tile_0_1)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.use_lock(%in__cons0__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.use_lock(%in__cons0__prodlock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.use_lock(%in__cons0__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.use_lock(%in__cons0__prodlock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.use_lock(%in__cons0__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.use_lock(%in__cons0__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.use_lock(%in__cons0__conslock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf0 : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.use_lock(%in__cons0__conslock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons0__buf1 : memref<32xi32> offset = 16 len = 16)
// CHECK-LABEL: aie.memtile_dma(%mem_tile_1_1)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.use_lock(%in__cons1__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.use_lock(%in__cons1__prodlock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.use_lock(%in__cons1__prodlock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.use_lock(%in__cons1__prodlock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.use_lock(%in__cons1__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.use_lock(%in__cons1__conslock0, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.use_lock(%in__cons1__conslock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf0 : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.use_lock(%in__cons1__conslock1, AcquireGreaterEqual
// CHECK-NEXT:  aie.dma_bd(%in__cons1__buf1 : memref<32xi32> offset = 16 len = 16)
module @two_distributes {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(0, 3)
    %t2 = aie.tile(1, 2)
    %t3 = aie.tile(1, 3)
    aie.objectfifo @in(%s, {%m0, %m1}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a0(%m0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a1(%m0, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@a0, @a1] ([] [0, 16])
    aie.objectfifo @b0(%m1, {%t2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%m1, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@b0, @b1] ([] [0, 16])
    aie.core(%t0) {
      %x = aie.objectfifo.acquire @a0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a0(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %x = aie.objectfifo.acquire @a1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a1(Consume, 1)
      aie.end
    }
    aie.core(%t2) {
      %x = aie.objectfifo.acquire @b0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b0(Consume, 1)
      aie.end
    }
    aie.core(%t3) {
      %x = aie.objectfifo.acquire @b1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b1(Consume, 1)
      aie.end
    }
  }
}
