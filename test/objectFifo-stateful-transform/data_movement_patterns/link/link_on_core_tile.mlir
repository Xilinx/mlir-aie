//===- link_on_core_tile.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// A compute tile distributing over its own DMAs: one S2MM chain writes both
// halves of each object, and each MM2S chain replays only the half its output
// carries, under that half's own lock pair.

module @distribute_on_core {
  aie.device(npu1) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile04 = aie.tile(0, 4)
    %tile05 = aie.tile(0, 5)

    aie.objectfifo @in (%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out0 (%tile02, {%tile04}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1 (%tile02, {%tile05}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out0, @out1] ([][0, 16])

    %core04 = aie.core(%tile04) {
      %e = aie.objectfifo.acquire @out0(Consume) : memref<16xi32>
      aie.objectfifo.release @out0(Consume) [1]
      aie.end
    }
    %core05 = aie.core(%tile05) {
      %e = aie.objectfifo.acquire @out1(Consume) : memref<16xi32>
      aie.objectfifo.release @out1(Consume) [1]
      aie.end
    }
  }
}

// One lock pair per half of the object.
// CHECK-DAG:   %[[PROD0:.*]] = aie.lock(%[[T02:.*]]) {init = 2 : i32, sym_name = "in_cons_prod_lock_0"}
// CHECK-DAG:   %[[CONS0:.*]] = aie.lock(%[[T02]]) {init = 0 : i32, sym_name = "in_cons_cons_lock_0"}
// CHECK-DAG:   %[[PROD1:.*]] = aie.lock(%[[T02]]) {init = 2 : i32, sym_name = "in_cons_prod_lock_1"}
// CHECK-DAG:   %[[CONS1:.*]] = aie.lock(%[[T02]]) {init = 0 : i32, sym_name = "in_cons_cons_lock_1"}

// Each output gets its own outgoing channel off the compute tile.
// CHECK-DAG:   aie.flow(%[[T02]], DMA : 0, %{{.*}}, DMA : 0)
// CHECK-DAG:   aie.flow(%[[T02]], DMA : 1, %{{.*}}, DMA : 0)

// CHECK:       aie.mem(%[[T02]])
// CHECK:         aie.dma_start(S2MM, 0
// CHECK:         aie.use_lock(%[[PROD0]], AcquireGreaterEqual
// CHECK:         aie.dma_bd(%[[BUF0:.*]] : memref<32xi32> offset = 0 len = 16)
// CHECK:         aie.use_lock(%[[CONS0]], Release
// CHECK:         aie.use_lock(%[[PROD1]], AcquireGreaterEqual
// CHECK:         aie.dma_bd(%[[BUF0]] : memref<32xi32> offset = 16 len = 16)
// CHECK:         aie.use_lock(%[[CONS1]], Release

// CHECK:         aie.dma_start(MM2S, 0
// CHECK:         aie.use_lock(%[[CONS0]], AcquireGreaterEqual
// CHECK:         aie.dma_bd(%[[BUF0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:         aie.use_lock(%[[PROD0]], Release

// CHECK:         aie.dma_start(MM2S, 1
// CHECK:         aie.use_lock(%[[CONS1]], AcquireGreaterEqual
// CHECK:         aie.dma_bd(%[[BUF0]] : memref<32xi32> offset = 16 len = 16)
// CHECK:         aie.use_lock(%[[PROD1]], Release
