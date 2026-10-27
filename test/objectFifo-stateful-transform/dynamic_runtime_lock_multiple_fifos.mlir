//===- dynamic_runtime_lock_multiple_fifos.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two objectFifos acquired/released with different counts in the same loop
// body. Each fifo gets its own runtime held counter and its own
// value-carrying `AcquireGreaterEqual`; the trailing post-loop releases
// decrement the respective counters.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK-DAG:           %[[MT:.*]] = aie.tile(0, 1)
// CHECK-DAG:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:           aie.buffer @[[YCB0:fifoY_cons_buff_0]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[YCB1:fifoY_cons_buff_1]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[YCB2:fifoY_cons_buff_2]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.lock @[[YCPROD:fifoY_cons_prod_lock_0]](%[[T2]]) {init = 3 : i32}
// CHECK-DAG:           aie.lock @[[YCCONS:fifoY_cons_cons_lock_0]](%[[T2]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[YB0:fifoY_buff_0]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[YB1:fifoY_buff_1]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.lock @[[YPROD:fifoY_prod_lock_0]](%[[MT]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[YCONS:fifoY_cons_lock_0]](%[[MT]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[XCB0:fifoX_cons_buff_0]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[XCB1:fifoX_cons_buff_1]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[XCB2:fifoX_cons_buff_2]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[XCB3:fifoX_cons_buff_3]](%[[T2]]) : memref<8xi8>
// CHECK-DAG:           aie.lock @[[XCPROD:fifoX_cons_prod_lock_0]](%[[T2]]) {init = 4 : i32}
// CHECK-DAG:           aie.lock @[[XCCONS:fifoX_cons_cons_lock_0]](%[[T2]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[XB0:fifoX_buff_0]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[XB1:fifoX_buff_1]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.buffer @[[XB2:fifoX_buff_2]](%[[MT]]) : memref<8xi8>
// CHECK-DAG:           aie.lock @[[XPROD:fifoX_prod_lock_0]](%[[MT]]) {init = 3 : i32}
// CHECK-DAG:           aie.lock @[[XCONS:fifoX_cons_lock_0]](%[[MT]]) {init = 0 : i32}
// CHECK-DAG:           aie.flow(%[[MT]], DMA : 0, %[[T2]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[MT]], DMA : 1, %[[T2]], DMA : 1)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C14:.*]] = arith.constant 14 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[C3I:.*]] = arith.constant 3 : i32
// CHECK:             %[[C2I:.*]] = arith.constant 2 : i32
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}}:2 = scf.for %{{.*}} = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[XH:.*]] = %[[C0I]], %[[YH:.*]] = %[[C0I]]) -> (i32, i32) {
// CHECK:               %[[XS:.*]] = arith.subi %[[C3I]], %[[XH]] : i32
// CHECK:               %[[XD:.*]] = arith.maxsi %[[XS]], %[[C0I]] : i32
// CHECK:               aie.use_lock(@[[XCCONS]], AcquireGreaterEqual, %[[XD]])
// CHECK:               %[[XNH:.*]] = arith.addi %[[XH]], %[[XD]] : i32
// CHECK:               %[[YS:.*]] = arith.subi %[[C2I]], %[[YH]] : i32
// CHECK:               %[[YD:.*]] = arith.maxsi %[[YS]], %[[C0I]] : i32
// CHECK:               aie.use_lock(@[[YCCONS]], AcquireGreaterEqual, %[[YD]])
// CHECK:               %[[YNH:.*]] = arith.addi %[[YH]], %[[YD]] : i32
// CHECK:               aie.use_lock(@[[XCPROD]], Release, %[[C1I]])
// CHECK:               %[[XRH:.*]] = arith.subi %[[XNH]], %[[C1I]] : i32
// CHECK:               aie.use_lock(@[[YCPROD]], Release, %[[C1I]])
// CHECK:               %[[YRH:.*]] = arith.subi %[[YNH]], %[[C1I]] : i32
// CHECK:               scf.yield %[[XRH]], %[[YRH]] : i32, i32
// CHECK:             }
// CHECK:             aie.use_lock(@[[XCPROD]], Release, %[[C2I]])
// CHECK:             aie.use_lock(@[[YCPROD]], Release, %[[C1I]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.memtile_dma(%[[MT]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(@[[YCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[YB0_ACCESS_0:.*]] = aie.get_buffer @[[YB0]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[YB0_ACCESS_0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[YPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(@[[YCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[YB1_ACCESS_1:.*]] = aie.get_buffer @[[YB1]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[YB1_ACCESS_1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[YPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(@[[XCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[XB0_ACCESS_2:.*]] = aie.get_buffer @[[XB0]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[XB0_ACCESS_2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[XPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(@[[XCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[XB1_ACCESS_3:.*]] = aie.get_buffer @[[XB1]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[XB1_ACCESS_3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[XPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(@[[XCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             %[[XB2_ACCESS_4:.*]] = aie.get_buffer @[[XB2]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[XB2_ACCESS_4]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[XPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(@[[YCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[YCB0_ACCESS_5:.*]] = aie.get_buffer @[[YCB0]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[YCB0_ACCESS_5]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[YCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(@[[YCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[YCB1_ACCESS_6:.*]] = aie.get_buffer @[[YCB1]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[YCB1_ACCESS_6]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[YCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(@[[YCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[YCB2_ACCESS_7:.*]] = aie.get_buffer @[[YCB2]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[YCB2_ACCESS_7]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[YCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 1, ^bb5, ^bb9)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(@[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[XCB0_ACCESS_8:.*]] = aie.get_buffer @[[XCB0]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[XCB0_ACCESS_8]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(@[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[XCB1_ACCESS_9:.*]] = aie.get_buffer @[[XCB1]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[XCB1_ACCESS_9]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(@[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[XCB2_ACCESS_10:.*]] = aie.get_buffer @[[XCB2]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[XCB2_ACCESS_10]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(@[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             %[[XCB3_ACCESS_11:.*]] = aie.get_buffer @[[XCB3]] : memref<8xi8>
// CHECK:             aie.dma_bd(%[[XCB3_ACCESS_11]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifoX(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @fifoY(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @fifoX(Consume, 3) : memref<8xi8>, memref<8xi8>, memref<8xi8>
        %y_obj0, %y_obj1 = aie.objectfifo.acquire @fifoY(Consume, 2) : memref<8xi8>, memref<8xi8>
        aie.objectfifo.release @fifoX(Consume, 1)
        aie.objectfifo.release @fifoY(Consume, 1)
      }
      aie.objectfifo.release @fifoX(Consume, 2)
      aie.objectfifo.release @fifoY(Consume, 1)
      aie.end
    }
  }
}
