//===- different_depth_objectfifos_test.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// The output fifo has depth [2, 2] and the input fifo depth [2, 3]. The consumer
// slides a 2-element window (acquire 2, release 1); the runtime lowering peels
// the first and last iterations and threads the buffer indices through the loop
// as iter_args. The input index selects among three buffers and wraps
// at 3, while the output index wraps at 2.

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           func.func @add_10_i32(%{{.*}}: memref<10xi32>, %{{.*}}: memref<10xi32>, %{{.*}}: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK-DAG:           %[[T0:.*]] = aie.tile(0, 0)
// CHECK-DAG:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:           %[[OF_B0:.*]] = aie.buffer(%[[T2]]) {sym_name = "output_fifo_buff_0"} : memref<10xi32>
// CHECK-DAG:           %[[OF_B1:.*]] = aie.buffer(%[[T2]]) {sym_name = "output_fifo_buff_1"} : memref<10xi32>
// CHECK-DAG:           %[[OF_PROD:.*]] = aie.lock(%[[T2]]) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK-DAG:           %[[OF_CONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK-DAG:           %[[IF_B0:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK-DAG:           %[[IF_B1:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32>
// CHECK-DAG:           %[[IF_B2:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_2"} : memref<10xi32>
// CHECK-DAG:           %[[IF_PROD:.*]] = aie.lock(%[[T2]]) {init = 3 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK-DAG:           %[[IF_CONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK-DAG:           aie.flow(%[[T0]], DMA : 0, %[[T2]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[T2]], DMA : 0, %[[T0]], DMA : 0)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[C9:.*]] = arith.constant 9 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[C2I:.*]] = arith.constant 2 : i32
// CHECK:             %[[C3I:.*]] = arith.constant 3 : i32
// CHECK:             aie.use_lock(%[[OF_PROD]], AcquireGreaterEqual, %[[C1I]])
// CHECK:             aie.use_lock(%[[IF_CONS]], AcquireGreaterEqual, %[[C1I]])
// CHECK:             func.call @add_10_i32(%[[IF_B0]], %[[IF_B0]], %[[OF_B0]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             aie.use_lock(%[[OF_CONS]], Release, %[[C1I]])
// CHECK:             %[[LOOP:.*]]:2 = scf.for %{{.*}} = %[[C0]] to %[[C9]] step %[[C1]] iter_args(%[[OIDX:.*]] = %[[C1I]], %[[IIDX:.*]] = %[[C0I]]) -> (i32, i32) {
// CHECK:               aie.use_lock(%[[OF_PROD]], AcquireGreaterEqual, %[[C1I]])
// CHECK:               %[[OC:.*]] = arith.index_cast %[[OIDX]] : i32 to index
// CHECK:               %[[OB:.*]] = scf.index_switch %[[OC]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[OF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               aie.use_lock(%[[IF_CONS]], AcquireGreaterEqual, %[[C1I]])
// CHECK:               %[[IC:.*]] = arith.index_cast %[[IIDX]] : i32 to index
// CHECK:               %[[IB0:.*]] = scf.index_switch %[[IC]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 scf.yield %[[IF_B2]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[IB1:.*]] = scf.index_switch %[[IC]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[IF_B2]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @add_10_i32(%[[IB0]], %[[IB1]], %[[OB]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(%[[IF_PROD]], Release, %[[C1I]])
// CHECK:               %[[IN:.*]] = arith.addi %[[IIDX]], %[[C1I]] : i32
// CHECK:               %[[ICMP:.*]] = arith.cmpi sge, %[[IN]], %[[C3I]] : i32
// CHECK:               %[[ISEL:.*]] = arith.select %[[ICMP]], %[[C0I]], %[[IN]] : i32
// CHECK:               aie.use_lock(%[[OF_CONS]], Release, %[[C1I]])
// CHECK:               %[[ON:.*]] = arith.addi %[[OIDX]], %[[C1I]] : i32
// CHECK:               %[[OCMP:.*]] = arith.cmpi sge, %[[ON]], %[[C2I]] : i32
// CHECK:               %[[OSEL:.*]] = arith.select %[[OCMP]], %[[C0I]], %[[ON]] : i32
// CHECK:               scf.yield %[[OSEL]], %[[ISEL]] : i32, i32
// CHECK:             }
// CHECK:             aie.use_lock(%[[OF_PROD]], AcquireGreaterEqual, %[[C1I]])
// CHECK:             %[[EOC:.*]] = arith.index_cast %[[LOOP]]#0 : i32 to index
// CHECK:             %[[EOB:.*]] = scf.index_switch %[[EOC]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[OF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[OF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             aie.use_lock(%[[IF_CONS]], AcquireGreaterEqual, %[[C1I]])
// CHECK:             %[[EIC:.*]] = arith.index_cast %[[LOOP]]#1 : i32 to index
// CHECK:             %[[EIB0:.*]] = scf.index_switch %[[EIC]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 2 {
// CHECK:               scf.yield %[[IF_B2]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[EIB1:.*]] = scf.index_switch %[[EIC]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[IF_B2]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 2 {
// CHECK:               scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[EIB0]], %[[EIB1]], %[[EOB]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             aie.use_lock(%[[IF_PROD]], Release, %[[C2I]])
// CHECK:             aie.use_lock(%[[OF_CONS]], Release, %[[C1I]])
// CHECK:             aie.end
// CHECK:           }
// CHECK-DAG:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[T0]], MM2S, 0)
// CHECK-DAG:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[T0]], S2MM, 0)
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[IF_B0]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[IF_B1]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[IF_B2]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF_CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[OF_B0]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[OF_PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[OF_CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[OF_B1]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[OF_PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }

module {
  aie.device(npu1_1col) {
    func.func @add_10_i32(%line_in1: memref<10xi32>, %line_in2: memref<10xi32>, %line_out: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, [2, 3]) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, [2, 2]) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 9 : index

      %1 = aie.objectfifo.acquire @output_fifo(Produce, 1) : memref<10xi32>
      %3 = aie.objectfifo.acquire @input_fifo(Consume, 1) : memref<10xi32>
      func.call @add_10_i32(%3, %3, %1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.objectfifo.release @output_fifo(Produce, 1)

      scf.for %arg0 = %c0 to %c8 step %c1 {
        %5 = aie.objectfifo.acquire @output_fifo(Produce, 1) : memref<10xi32>
        %7, %8 = aie.objectfifo.acquire @input_fifo(Consume, 2) : memref<10xi32>, memref<10xi32>
        func.call @add_10_i32(%7, %8, %5) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
        aie.objectfifo.release @output_fifo(Produce, 1)
      }

      %10 = aie.objectfifo.acquire @output_fifo(Produce, 1) : memref<10xi32>
      %12, %13 = aie.objectfifo.acquire @input_fifo(Consume, 2) : memref<10xi32>, memref<10xi32>
      func.call @add_10_i32(%12, %13, %10) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.objectfifo.release @input_fifo(Consume, 2)
      aie.objectfifo.release @output_fifo(Produce, 1)

      aie.end
    }
  }
}
