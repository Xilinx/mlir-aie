//===- link_join_unequal_repeat_count.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s --check-prefix=POOL
// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s --check-prefix=DMA

// One A object repeated twice joins two independently produced B objects.
// POOL-LABEL: module @join_unequal
// POOL: aie.objectfifo.pool @a__pool
// POOL-SAME: repeatCount = 2
// POOL: aie.objectfifo.pool @b__pool
// POOL-NOT: repeatCount
// POOL: aie.objectfifo.pool @out__pool
// POOL-NOT: repeatCount
// POOL: aie.objectfifo.segment @s0
// POOL: aie.objectfifo.segment @s1
// DMA-LABEL: module @join_unequal
// DMA: %[[AP:.*]] = aie.lock({{.*}}) {init = 2 : i32, sym_name = "a__prodlock0"}
// DMA: %[[AC:.*]] = aie.lock({{.*}}) {init = 0 : i32, sym_name = "a__conslock0"}
// DMA: %[[BP:.*]] = aie.lock({{.*}}) {init = 1 : i32, sym_name = "b__prodlock0"}
// DMA: %[[BC:.*]] = aie.lock({{.*}}) {init = 0 : i32, sym_name = "b__conslock0"}
// DMA: %[[OP0:.*]] = aie.lock({{.*}}) {init = 1 : i32, sym_name = "out__prodlock0"}
// DMA: %[[OC0:.*]] = aie.lock({{.*}}) {init = 0 : i32, sym_name = "out__conslock0"}
// DMA: %[[OP1:.*]] = aie.lock({{.*}}) {init = 1 : i32, sym_name = "out__prodlock1"}
// DMA: %[[OC1:.*]] = aie.lock({{.*}}) {init = 0 : i32, sym_name = "out__conslock1"}
// DMA: aie.core(
// DMA: %[[TWO:.*]] = arith.constant 2 : i32
// DMA: aie.use_lock(%[[AP]], AcquireGreaterEqual, %[[TWO]])
// DMA: aie.use_lock(%[[AC]], Release, %[[TWO]])
// DMA: aie.core(
// DMA: %[[ONE:.*]] = arith.constant 1 : i32
// DMA: aie.use_lock(%[[BP]], AcquireGreaterEqual, %[[ONE]])
// DMA: aie.use_lock(%[[BC]], Release, %[[ONE]])
// DMA: aie.use_lock(%[[BP]], AcquireGreaterEqual, %[[ONE]])
// DMA: aie.use_lock(%[[BC]], Release, %[[ONE]])
// DMA: aie.mem(
// DMA: aie.dma_start(MM2S, 0, {{.*}}, repeat_count = 1)
// DMA: aie.memtile_dma(
// DMA: %[[UNIT:.*]] = arith.constant 1 : i32
// DMA: aie.dma_start(S2MM, 0,
// DMA: aie.use_lock(%[[OP0]], AcquireGreaterEqual, %[[UNIT]])
// DMA: aie.use_lock(%[[OC0]], Release, %[[UNIT]])
// DMA: aie.dma_start(S2MM, 1,
// DMA: aie.use_lock(%[[OP1]], AcquireGreaterEqual, %[[UNIT]])
// DMA: aie.use_lock(%[[OC1]], Release, %[[UNIT]])
// DMA: aie.dma_start(MM2S, 0,
// DMA: aie.use_lock(%[[OC0]], AcquireGreaterEqual, %[[UNIT]])
// DMA: aie.use_lock(%[[OP0]], Release, %[[UNIT]])
// DMA: aie.use_lock(%[[OC1]], AcquireGreaterEqual, %[[UNIT]])
// DMA: aie.use_lock(%[[OP1]], Release, %[[UNIT]])
module @join_unequal {
  aie.device(npu1) {
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    %link = aie.tile(1, 1)
    %out = aie.tile(1, 4)
    aie.objectfifo @a(%a, {%link}, 1 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%b, {%link}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%link, {%out}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 16] [])
    %ca = aie.core(%a) {
      %ea = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %cb = aie.core(%b) {
      %eb = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      %eb2 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    %co = aie.core(%out) {
      %eo = aie.objectfifo.acquire @out(Consume, 1) : memref<32xi32>
      aie.objectfifo.release @out(Consume, 1)
      %eo2 = aie.objectfifo.acquire @out(Consume, 1) : memref<32xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
  }
}
