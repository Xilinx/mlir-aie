//===- dynamic_runtime_lock_conditional.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An acquire in the loop body with a release nested in an `scf.if`. The
// runtime held-counter bookkeeping handles conditional acquire/release with
// no static analysis: the held decrement and the buffer-index update are
// emitted inside the `scf.if`, so they only run when the branch is taken.
// The next iteration's acquire computes its delta from the current held
// value, so it acquires nothing extra when the release did not fire.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK-DAG:           %[[MT:.*]] = aie.tile(0, 1)
// CHECK-DAG:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:           aie.buffer @[[CB0:fifo_cons_buff_0]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[CB1:fifo_cons_buff_1]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[CB2:fifo_cons_buff_2]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[CB3:fifo_cons_buff_3]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.lock @[[CPROD:fifo_cons_prod_lock_0]](%[[T2]]) {init = 4 : i32}
// CHECK-DAG:           aie.lock @[[CCONS:fifo_cons_cons_lock_0]](%[[T2]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[B0:fifo_buff_0]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[B1:fifo_buff_1]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[B2:fifo_buff_2]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.lock @[[PROD:fifo_prod_lock_0]](%[[MT]]) {init = 3 : i32}
// CHECK-DAG:           aie.lock @[[CONS:fifo_cons_lock_0]](%[[MT]]) {init = 0 : i32}
// CHECK-DAG:           aie.flow(%[[MT]], DMA : 0, %[[T2]], DMA : 0)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C14:.*]] = arith.constant 14 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[C3I:.*]] = arith.constant 3 : i32
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = scf.for %{{.*}} = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[HELD:.*]] = %[[C0I]]) -> (i32) {
// CHECK:               %[[SUB:.*]] = arith.subi %[[C3I]], %[[HELD]] : i32
// CHECK:               %[[DELTA:.*]] = arith.maxsi %[[SUB]], %[[C0I]] : i32
// CHECK:               aie.use_lock(@[[CCONS]], AcquireGreaterEqual, %[[DELTA]])
// CHECK:               %[[NH:.*]] = arith.addi %[[HELD]], %[[DELTA]] : i32
// CHECK:               aie.use_lock(@[[CPROD]], Release, %[[C1I]])
// CHECK:               %[[RH:.*]] = arith.subi %[[NH]], %[[C1I]] : i32
// CHECK:               scf.yield %[[RH]] : i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.memtile_dma(%[[MT]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(@[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[B0_ACCESS_0:.*]] = aie.get_buffer @[[B0]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[B0_ACCESS_0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(@[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[B1_ACCESS_1:.*]] = aie.get_buffer @[[B1]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[B1_ACCESS_1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(@[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[B2_ACCESS_2:.*]] = aie.get_buffer @[[B2]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[B2_ACCESS_2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(@[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[CB0_ACCESS_3:.*]] = aie.get_buffer @[[CB0]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[CB0_ACCESS_3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(@[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[CB1_ACCESS_4:.*]] = aie.get_buffer @[[CB1]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[CB1_ACCESS_4]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(@[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[CB2_ACCESS_5:.*]] = aie.get_buffer @[[CB2]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[CB2_ACCESS_5]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(@[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[CB3_ACCESS_6:.*]] = aie.get_buffer @[[CB3]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[CB3_ACCESS_6]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %true = arith.constant true
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @fifo(Consume, 3) : memref<8xi8>, memref<8xi8>, memref<8xi8>
        scf.if %true {
          aie.objectfifo.release @fifo(Consume, 1)
        }
      }
      aie.end
    }
  }
}
