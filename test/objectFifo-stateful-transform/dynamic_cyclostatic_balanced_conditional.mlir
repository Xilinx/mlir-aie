//===- dynamic_cyclostatic_balanced_conditional.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for the multi-head-attention lowering (see commit that
// stabilized cyclostatic pattern analysis). A (fifo, port) is used BOTH
// unconditionally AND inside an scf.if, but every occurrence is *balanced*
// (acquire count == release count within its scope). A balanced conditional
// branch contributes zero net carry, so it cannot make the straight-line
// carry analysis unsound: the pass must NOT emit the "cannot statically
// analyze cyclostatic acquire pattern" diagnostic, must NOT peel, and must
// lower normally.
//
// Before the fix, the mere co-occurrence of conditional + unconditional
// acq/rel on the same fifo tripped a hard error even though the program is
// well-formed.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_5]]) {init = 2 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:           %[[VAL_10:.*]] = aie.core(%[[VAL_5]]) {
// CHECK:             %[[VAL_11:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[VAL_15:.*]] = %[[VAL_13]] to %[[VAL_11]] step %[[VAL_12]] {
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:               aie.use_lock(%[[VAL_8]], Release, %[[VAL_14]])
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:               aie.use_lock(%[[VAL_8]], Release, %[[VAL_14]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_17:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_18:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_17]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_17]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = aie.mem(%[[VAL_5]]) {
// CHECK:             %[[VAL_20:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_21:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %true = arith.constant true
      scf.for %arg0 = %c0 to %c14 step %c1 {
        // Unconditional, balanced: acquire 1, release 1 -> net 0.
        %a_obj0 = aie.objectfifo.acquire @fifo(Consume) : memref<8xi8>
        aie.objectfifo.release @fifo(Consume) [1]
        // Conditional, balanced: acquire 1, release 1 -> net 0 per branch.
        scf.if %true {
          %b_obj0 = aie.objectfifo.acquire @fifo(Consume) : memref<8xi8>
          aie.objectfifo.release @fifo(Consume) [1]
        }
      }
      aie.end
    }
  }
}
