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

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[MT:.*]] = aie.tile(0, 1)
// CHECK:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK:           %[[YCB0:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifoY_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[YCB1:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifoY_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[YCB2:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifoY_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[YCPROD:.*]] = aie.lock(%[[T2]]) {init = 3 : i32, sym_name = "fifoY_cons_prod_lock_0"}
// CHECK:           %[[YCCONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "fifoY_cons_cons_lock_0"}
// CHECK:           %[[YB0:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifoY_buff_0"} : memref<8xi8>
// CHECK:           %[[YB1:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifoY_buff_1"} : memref<8xi8>
// CHECK:           %[[YPROD:.*]] = aie.lock(%[[MT]]) {init = 2 : i32, sym_name = "fifoY_prod_lock_0"}
// CHECK:           %[[YCONS:.*]] = aie.lock(%[[MT]]) {init = 0 : i32, sym_name = "fifoY_cons_lock_0"}
// CHECK:           %[[XCB0:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifoX_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[XCB1:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifoX_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[XCB2:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifoX_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[XCB3:.*]] = aie.buffer(%[[T2]]) {sym_name = "fifoX_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[XCPROD:.*]] = aie.lock(%[[T2]]) {init = 4 : i32, sym_name = "fifoX_cons_prod_lock_0"}
// CHECK:           %[[XCCONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "fifoX_cons_cons_lock_0"}
// CHECK:           %[[XB0:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifoX_buff_0"} : memref<8xi8>
// CHECK:           %[[XB1:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifoX_buff_1"} : memref<8xi8>
// CHECK:           %[[XB2:.*]] = aie.buffer(%[[MT]]) {sym_name = "fifoX_buff_2"} : memref<8xi8>
// CHECK:           %[[XPROD:.*]] = aie.lock(%[[MT]]) {init = 3 : i32, sym_name = "fifoX_prod_lock_0"}
// CHECK:           %[[XCONS:.*]] = aie.lock(%[[MT]]) {init = 0 : i32, sym_name = "fifoX_cons_lock_0"}
// CHECK:           aie.flow(%[[MT]], DMA : 0, %[[T2]], DMA : 0)
// CHECK:           aie.flow(%[[MT]], DMA : 1, %[[T2]], DMA : 1)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C14:.*]] = arith.constant 14 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[C3I:.*]] = arith.constant 3 : i32
// CHECK:             %[[C2I:.*]] = arith.constant 2 : i32
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[C4I:.*]] = arith.constant 4 : i32
// CHECK:             %{{.*}}:4 = scf.for %{{.*}} = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[XI:.*]] = %[[C0I]], %[[YI:.*]] = %[[C0I]], %[[XH:.*]] = %[[C0I]], %[[YH:.*]] = %[[C0I]]) -> (i32, i32, i32, i32) {
// CHECK:               %[[XS:.*]] = arith.subi %[[C3I]], %[[XH]] : i32
// CHECK:               %[[XD:.*]] = arith.maxsi %[[XS]], %[[C0I]] : i32
// CHECK:               aie.use_lock(%[[XCCONS]], AcquireGreaterEqual, %[[XD]])
// CHECK:               %[[XNH:.*]] = arith.addi %[[XH]], %[[XD]] : i32
// CHECK:               %[[YS:.*]] = arith.subi %[[C2I]], %[[YH]] : i32
// CHECK:               %[[YD:.*]] = arith.maxsi %[[YS]], %[[C0I]] : i32
// CHECK:               aie.use_lock(%[[YCCONS]], AcquireGreaterEqual, %[[YD]])
// CHECK:               %[[YNH:.*]] = arith.addi %[[YH]], %[[YD]] : i32
// CHECK:               aie.use_lock(%[[XCPROD]], Release, %[[C1I]])
// CHECK:               %[[XRH:.*]] = arith.subi %[[XNH]], %[[C1I]] : i32
// CHECK:               %[[XNX:.*]] = arith.addi %[[XI]], %[[C1I]] : i32
// CHECK:               %[[XCMP:.*]] = arith.cmpi sge, %[[XNX]], %[[C4I]] : i32
// CHECK:               %[[XSEL:.*]] = arith.select %[[XCMP]], %[[C0I]], %[[XNX]] : i32
// CHECK:               aie.use_lock(%[[YCPROD]], Release, %[[C1I]])
// CHECK:               %[[YRH:.*]] = arith.subi %[[YNH]], %[[C1I]] : i32
// CHECK:               %[[YNX:.*]] = arith.addi %[[YI]], %[[C1I]] : i32
// CHECK:               %[[YCMP:.*]] = arith.cmpi sge, %[[YNX]], %[[C3I]] : i32
// CHECK:               %[[YSEL:.*]] = arith.select %[[YCMP]], %[[C0I]], %[[YNX]] : i32
// CHECK:               scf.yield %[[XSEL]], %[[YSEL]], %[[XRH]], %[[YRH]] : i32, i32, i32, i32
// CHECK:             }
// CHECK:             aie.use_lock(%[[XCPROD]], Release, %[[C2I]])
// CHECK:             aie.use_lock(%[[YCPROD]], Release, %[[C1I]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.memtile_dma(%[[MT]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[XCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[XB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[XCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[XB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[XCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[XB2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 1, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[YCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[YB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[YPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[YCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[YB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[YPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[XCB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[XCB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[XCB2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[XCB3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 1, ^bb6, ^bb9)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[YCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[YCB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[YCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[YCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[YCB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[YCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[YCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[YCB2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[YCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb6
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
        %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @fifoX(Consume) : memref<8xi8>, memref<8xi8>, memref<8xi8>
        %y_obj0, %y_obj1 = aie.objectfifo.acquire @fifoY(Consume) : memref<8xi8>, memref<8xi8>
        aie.objectfifo.release @fifoX(Consume) [1]
        aie.objectfifo.release @fifoY(Consume) [1]
      }
      aie.objectfifo.release @fifoX(Consume) [2]
      aie.objectfifo.release @fifoY(Consume) [1]
      aie.end
    }
  }
}
