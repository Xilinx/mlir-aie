//===- dynamic_runtime_lock_nested_loops.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Acquires/releases at both the outer and inner loop levels on two fifos.
// The runtime held counters are shared across the loop nest, so the inner
// loop's per-iteration acquire delta and the outer loop's acquire delta are
// each computed from the current held value regardless of loop depth.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK-DAG:           %[[MT:.*]] = aie.tile(0, 1)
// CHECK-DAG:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:           %[[XCB0:.*]] = aie.buffer(%[[T2]]) {sym_name = "inOF_X_cons_buff_0"} : memref<8xi8>
// CHECK-DAG:           %[[XCB1:.*]] = aie.buffer(%[[T2]]) {sym_name = "inOF_X_cons_buff_1"} : memref<8xi8>
// CHECK-DAG:           %[[XCB2:.*]] = aie.buffer(%[[T2]]) {sym_name = "inOF_X_cons_buff_2"} : memref<8xi8>
// CHECK-DAG:           %[[XCB3:.*]] = aie.buffer(%[[T2]]) {sym_name = "inOF_X_cons_buff_3"} : memref<8xi8>
// CHECK-DAG:           %[[XCPROD:.*]] = aie.lock(%[[T2]]) {init = 4 : i32, sym_name = "inOF_X_cons_prod_lock_0"}
// CHECK-DAG:           %[[XCCONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "inOF_X_cons_cons_lock_0"}
// CHECK-DAG:           %[[XB0:.*]] = aie.buffer(%[[MT]]) {sym_name = "inOF_X_buff_0"} : memref<8xi8>
// CHECK-DAG:           %[[XB1:.*]] = aie.buffer(%[[MT]]) {sym_name = "inOF_X_buff_1"} : memref<8xi8>
// CHECK-DAG:           %[[XB2:.*]] = aie.buffer(%[[MT]]) {sym_name = "inOF_X_buff_2"} : memref<8xi8>
// CHECK-DAG:           %[[XPROD:.*]] = aie.lock(%[[MT]]) {init = 3 : i32, sym_name = "inOF_X_prod_lock_0"}
// CHECK-DAG:           %[[XCONS:.*]] = aie.lock(%[[MT]]) {init = 0 : i32, sym_name = "inOF_X_cons_lock_0"}
// CHECK-DAG:           %[[WCB0:.*]] = aie.buffer(%[[T2]]) {sym_name = "inOF_W_cons_buff_0"} : memref<8xi8>
// CHECK-DAG:           %[[WCB1:.*]] = aie.buffer(%[[T2]]) {sym_name = "inOF_W_cons_buff_1"} : memref<8xi8>
// CHECK-DAG:           %[[WCB2:.*]] = aie.buffer(%[[T2]]) {sym_name = "inOF_W_cons_buff_2"} : memref<8xi8>
// CHECK-DAG:           %[[WCPROD:.*]] = aie.lock(%[[T2]]) {init = 3 : i32, sym_name = "inOF_W_cons_prod_lock_0"}
// CHECK-DAG:           %[[WCCONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "inOF_W_cons_cons_lock_0"}
// CHECK-DAG:           %[[WB0:.*]] = aie.buffer(%[[MT]]) {sym_name = "inOF_W_buff_0"} : memref<8xi8>
// CHECK-DAG:           %[[WB1:.*]] = aie.buffer(%[[MT]]) {sym_name = "inOF_W_buff_1"} : memref<8xi8>
// CHECK-DAG:           %[[WB2:.*]] = aie.buffer(%[[MT]]) {sym_name = "inOF_W_buff_2"} : memref<8xi8>
// CHECK-DAG:           %[[WPROD:.*]] = aie.lock(%[[MT]]) {init = 3 : i32, sym_name = "inOF_W_prod_lock_0"}
// CHECK-DAG:           %[[WCONS:.*]] = aie.lock(%[[MT]]) {init = 0 : i32, sym_name = "inOF_W_cons_lock_0"}
// CHECK-DAG:           aie.flow(%[[MT]], DMA : 0, %[[T2]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[MT]], DMA : 1, %[[T2]], DMA : 1)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C14:.*]] = arith.constant 14 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[C2I:.*]] = arith.constant 2 : i32
// CHECK:             %[[C3I:.*]] = arith.constant 3 : i32
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[C4I:.*]] = arith.constant 4 : i32
// CHECK:             %[[NEG2:.*]] = arith.constant -2 : i32
// CHECK:             %{{.*}}:4 = scf.for %{{.*}} = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[WI:.*]] = %[[C0I]], %[[XI:.*]] = %[[C0I]], %[[WH:.*]] = %[[C0I]], %[[XH:.*]] = %[[C0I]]) -> (i32, i32, i32, i32) {
// CHECK:               %[[WS:.*]] = arith.subi %[[C2I]], %[[WH]] : i32
// CHECK:               %[[WD:.*]] = arith.maxsi %[[WS]], %[[C0I]] : i32
// CHECK:               aie.use_lock(%[[WCCONS]], AcquireGreaterEqual, %[[WD]])
// CHECK:               %[[WNH:.*]] = arith.addi %[[WH]], %[[WD]] : i32
// CHECK:               %[[INNER:.*]]:2 = scf.for %{{.*}} = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[IXI:.*]] = %[[XI]], %[[IXH:.*]] = %[[XH]]) -> (i32, i32) {
// CHECK:                 %[[XS:.*]] = arith.subi %[[C3I]], %[[IXH]] : i32
// CHECK:                 %[[XD:.*]] = arith.maxsi %[[XS]], %[[C0I]] : i32
// CHECK:                 aie.use_lock(%[[XCCONS]], AcquireGreaterEqual, %[[XD]])
// CHECK:                 %[[XNH:.*]] = arith.addi %[[IXH]], %[[XD]] : i32
// CHECK:                 aie.use_lock(%[[XCPROD]], Release, %[[C1I]])
// CHECK:                 %[[XRH:.*]] = arith.subi %[[XNH]], %[[C1I]] : i32
// CHECK:                 %[[XNX:.*]] = arith.addi %[[IXI]], %[[C1I]] : i32
// CHECK:                 %[[XCMP:.*]] = arith.cmpi sge, %[[XNX]], %[[C4I]] : i32
// CHECK:                 %[[XSEL:.*]] = arith.select %[[XCMP]], %[[C0I]], %[[XNX]] : i32
// CHECK:                 scf.yield %[[XSEL]], %[[XRH]] : i32, i32
// CHECK:               }
// CHECK:               aie.use_lock(%[[XCPROD]], Release, %[[C2I]])
// CHECK:               %[[XRH2:.*]] = arith.subi %[[INNER]]#1, %[[C2I]] : i32
// CHECK:               %[[XNX2:.*]] = arith.addi %[[INNER]]#0, %[[C2I]] : i32
// CHECK:               %[[XCMP2:.*]] = arith.cmpi sge, %[[XNX2]], %[[C4I]] : i32
// CHECK:               %[[XWR2:.*]] = arith.addi %[[INNER]]#0, %[[NEG2]] : i32
// CHECK:               %[[XSEL2:.*]] = arith.select %[[XCMP2]], %[[XWR2]], %[[XNX2]] : i32
// CHECK:               aie.use_lock(%[[WCPROD]], Release, %[[C1I]])
// CHECK:               %[[WRH:.*]] = arith.subi %[[WNH]], %[[C1I]] : i32
// CHECK:               %[[WNX:.*]] = arith.addi %[[WI]], %[[C1I]] : i32
// CHECK:               %[[WCMP:.*]] = arith.cmpi sge, %[[WNX]], %[[C3I]] : i32
// CHECK:               %[[WSEL:.*]] = arith.select %[[WCMP]], %[[C0I]], %[[WNX]] : i32
// CHECK:               scf.yield %[[WSEL]], %[[XSEL2]], %[[WRH]], %[[XRH2]] : i32, i32, i32, i32
// CHECK:             }
// CHECK:             aie.use_lock(%[[WCPROD]], Release, %[[C1I]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.memtile_dma(%[[MT]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[WCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[WB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[WPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[WCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[WB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[WPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[WCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[WB2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[WPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 1, ^bb5, ^bb8)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[XCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[XB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[XCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[XB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[XCONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[XB2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XPROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[WCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[WCB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[WCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[WCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[WCB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[WCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[WCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[WCB2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[WCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 1, ^bb5, ^bb9)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[XCB0]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[XCB1]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[XCB2]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[XCPROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[XCB3]] : memref<8xi8> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[XCCONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @inOF_W(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @inOF_X(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %w_obj0, %w_obj1 = aie.objectfifo.acquire @inOF_W(Consume, 2) : memref<8xi8>, memref<8xi8>
        scf.for %arg1 = %c0 to %c14 step %c1 {
          %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @inOF_X(Consume, 3) : memref<8xi8>, memref<8xi8>, memref<8xi8>
          aie.objectfifo.release @inOF_X(Consume, 1)
        }
        aie.objectfifo.release @inOF_X(Consume, 2)
        aie.objectfifo.release @inOF_W(Consume, 1)
      }
      aie.objectfifo.release @inOF_W(Consume, 1)
      aie.end
    }
  }
}
