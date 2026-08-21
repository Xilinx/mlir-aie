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
// CHECK-DAG:           %[[MT:.*]] = aie.tile(0, 1)
// CHECK-DAG:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:           %[[CB0:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK-DAG:           %[[CB1:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK-DAG:           %[[CPROD:.*]] = aie.lock(%[[T2]]) {init = 2 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK-DAG:           %[[CCONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK-DAG:           %[[B0:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK-DAG:           %[[B1:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK-DAG:           %[[PROD:.*]] = aie.lock(%[[MT]]) {init = 2 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK-DAG:           %[[CONS:.*]] = aie.lock(%[[MT]]) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK-DAG:           aie.flow(%[[MT]], DMA : 0, %[[T2]], DMA : 0)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C14:.*]] = arith.constant 14 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %{{.*}} = %[[C0]] to %[[C14]] step %[[C1]] {
// CHECK:               aie.use_lock(%[[CCONS]], AcquireGreaterEqual, %[[C1I]])
// CHECK:               aie.use_lock(%[[CPROD]], Release, %[[C1I]])
// CHECK:               aie.use_lock(%[[CCONS]], AcquireGreaterEqual, %[[C1I]])
// CHECK:               aie.use_lock(%[[CPROD]], Release, %[[C1I]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.memtile_dma(%[[MT]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
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
