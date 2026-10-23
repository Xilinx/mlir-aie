//===- link_independent_repeat_count.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split %s | FileCheck %s --check-prefix=POOL
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s --check-prefix=DMA

// Input replication and output replication are separate stages. The join's
// shared pool must retain the output repeat, not either producer's repeat.
// POOL-LABEL: module @join_independent
// POOL: aie.objectfifo.pool @a_pool
// POOL-SAME: repeatCount = 3
// POOL: aie.objectfifo.pool @b_pool
// POOL-SAME: repeatCount = 3
// POOL: aie.objectfifo.pool @out_pool
// POOL-SAME: repeatCount = 2
// DMA-LABEL: module @join_independent
// DMA: aie.mem(
// DMA: aie.dma_start(MM2S, 0, {{.*}}, repeat_count = 2)
// DMA: aie.memtile_dma(
// DMA: %[[TWO:.*]] = arith.constant 2 : i32
// DMA: aie.use_lock({{.*}}, AcquireGreaterEqual, %[[TWO]])
// DMA: aie.dma_start(MM2S, 0,
// DMA-COUNT-4: aie.dma_bd(
// DMA-NOT: aie.dma_bd(
// DMA: aie.mem(
// DMA: aie.dma_start(MM2S, 0, {{.*}}, repeat_count = 2)
module @join_independent {
  aie.device(npu1) {
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    %link = aie.tile(1, 1)
    %out = aie.tile(1, 0)
    aie.objectfifo @a(%a, {%link}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%b, {%link}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%link, {%out}, 1 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 16] [])
    %ca = aie.core(%a) {
      %ea = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %cb = aie.core(%b) {
      %eb = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
  }
}

// -----

// A distribute keeps the input's local repeat separate from the repeat of
// the shared pool feeding the two output segments.
// POOL-LABEL: module @distribute_independent
// POOL: aie.objectfifo.pool @in_cons_pool
// POOL-SAME: repeatCount = 3
// POOL: aie.objectfifo.pool @in_pool
// POOL-SAME: repeatCount = 2
// DMA-LABEL: module @distribute_independent
// DMA: aie.mem(
// DMA: aie.dma_start(MM2S, 0, {{.*}}, repeat_count = 1)
// DMA: aie.memtile_dma(
// DMA: %[[THREE:.*]] = arith.constant 3 : i32
// DMA: aie.use_lock({{.*}}, AcquireGreaterEqual, %[[THREE]])
// DMA: aie.dma_start(MM2S, 0, {{.*}}, repeat_count = 2)
// DMA: aie.dma_start(MM2S, 1, {{.*}}, repeat_count = 2)
module @distribute_independent {
  aie.device(npu1) {
    %in = aie.tile(2, 2)
    %link = aie.tile(1, 1)
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    aie.objectfifo @in(%in, {%link}, 1 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%link, {%a}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%link, {%b}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@a, @b] ([] [0, 16])
    %ci = aie.core(%in) {
      %ei = aie.objectfifo.acquire @in(Produce, 1) : memref<32xi32>
      aie.objectfifo.release @in(Produce, 1)
      aie.end
    }
    %ca = aie.core(%a) {
      %ea = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    %cb = aie.core(%b) {
      %eb = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
