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
// CHECK-DAG:           %[[CB0:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK-DAG:           %[[CB1:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK-DAG:           %[[CB2:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_2"} : memref<8xi8>
// CHECK-DAG:           %[[CB3:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifo_cons_buff_3"} : memref<8xi8>
// CHECK-DAG:           %[[CPROD:.*]] = aie.lock(%[[T2]]) {init = 4 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK-DAG:           %[[CCONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK-DAG:           %[[B0:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK-DAG:           %[[B1:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK-DAG:           %[[B2:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK-DAG:           %[[PROD:.*]] = aie.lock(%[[MT]]) {init = 3 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK-DAG:           %[[CONS:.*]] = aie.lock(%[[MT]]) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK-DAG:           aie.flow(%[[MT]], DMA : 0, %[[T2]], DMA : 0)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C14:.*]] = arith.constant 14 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[C3I:.*]] = arith.constant 3 : i32
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[C4I:.*]] = arith.constant 4 : i32
// CHECK:             %{{.*}}:2 = scf.for %{{.*}} = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[IDX:.*]] = %[[C0I]], %[[HELD:.*]] = %[[C0I]]) -> (i32, i32) {
// CHECK:               %[[SUB:.*]] = arith.subi %[[C3I]], %[[HELD]] : i32
// CHECK:               %[[DELTA:.*]] = arith.maxsi %[[SUB]], %[[C0I]] : i32
// CHECK:               aie.use_lock(%[[CCONS]], AcquireGreaterEqual, %[[DELTA]])
// CHECK:               %[[NH:.*]] = arith.addi %[[HELD]], %[[DELTA]] : i32
// CHECK:               aie.use_lock(%[[CPROD]], Release, %[[C1I]])
// CHECK:               %[[RH:.*]] = arith.subi %[[NH]], %[[C1I]] : i32
// CHECK:               %[[NX:.*]] = arith.addi %[[IDX]], %[[C1I]] : i32
// CHECK:               %[[CMP:.*]] = arith.cmpi sge, %[[NX]], %[[C4I]] : i32
// CHECK:               %[[SEL:.*]] = arith.select %[[CMP]], %[[C0I]], %[[NX]] : i32
// CHECK:               scf.yield %[[SEL]], %[[RH]] : i32, i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.memtile_dma(%[[MT]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[B2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[CPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[CB3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[CCONS]], Release, %[[N1]])
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
        %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @fifo (Consume, 3) : memref<8xi8>, memref<8xi8>, memref<8xi8>
        scf.if %true {
          aie.objectfifo.release @fifo (Consume, 1)
        }
      }
      aie.end
    }
  }
}
