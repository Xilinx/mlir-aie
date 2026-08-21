//===- core_flag_test.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// Per-core `dynamic_objfifo_lowering` flag: core_0_2 is flagged, so its loop
// stays rolled with runtime bookkeeping (iter_args + index_switch). core_0_4 is
// unflagged, so it is statically unrolled (fixed buffers, no index_switch).

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           func.func @passthrough_10_i32(%[[VAL_0:.*]]: memref<10xi32>, %[[VAL_1:.*]]: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK-DAG:           %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK-DAG:           %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:           %[[T4:.*]] = aie.tile(0, 4)
// CHECK-DAG:           %{{.*}} = aie.lock(%[[SHIM]]) {init = 0 : i32, sym_name = "output_fifo2_cons_prod_lock_0"}
// CHECK-DAG:           %{{.*}} = aie.lock(%[[SHIM]]) {init = 0 : i32, sym_name = "output_fifo2_cons_cons_lock_0"}
// CHECK-DAG:           %[[OF2_B0:.*]] = aie.buffer(%[[T4]]) {sym_name = "output_fifo2_buff_0"} : memref<10xi32>
// CHECK-DAG:           %[[OF2_B1:.*]] = aie.buffer(%[[T4]]) {sym_name = "output_fifo2_buff_1"} : memref<10xi32>
// CHECK-DAG:           %[[OF2_PROD:.*]] = aie.lock(%[[T4]]) {init = 2 : i32, sym_name = "output_fifo2_prod_lock_0"}
// CHECK-DAG:           %[[OF2_CONS:.*]] = aie.lock(%[[T4]]) {init = 0 : i32, sym_name = "output_fifo2_cons_lock_0"}
// CHECK-DAG:           %[[IF2_B0:.*]] = aie.buffer(%[[T4]]) {sym_name = "input_fifo2_cons_buff_0"} : memref<10xi32>
// CHECK-DAG:           %[[IF2_B1:.*]] = aie.buffer(%[[T4]]) {sym_name = "input_fifo2_cons_buff_1"} : memref<10xi32>
// CHECK-DAG:           %[[IF2_PROD:.*]] = aie.lock(%[[T4]]) {init = 2 : i32, sym_name = "input_fifo2_cons_prod_lock_0"}
// CHECK-DAG:           %[[IF2_CONS:.*]] = aie.lock(%[[T4]]) {init = 0 : i32, sym_name = "input_fifo2_cons_cons_lock_0"}
// CHECK-DAG:           %{{.*}} = aie.lock(%[[SHIM]]) {init = 0 : i32, sym_name = "input_fifo2_prod_lock_0"}
// CHECK-DAG:           %{{.*}} = aie.lock(%[[SHIM]]) {init = 0 : i32, sym_name = "input_fifo2_cons_lock_0"}
// CHECK-DAG:           %{{.*}} = aie.lock(%[[SHIM]]) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock_0"}
// CHECK-DAG:           %{{.*}} = aie.lock(%[[SHIM]]) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock_0"}
// CHECK-DAG:           %[[OF_B0:.*]] = aie.buffer(%[[T2]]) {sym_name = "output_fifo_buff_0"} : memref<10xi32>
// CHECK-DAG:           %[[OF_B1:.*]] = aie.buffer(%[[T2]]) {sym_name = "output_fifo_buff_1"} : memref<10xi32>
// CHECK-DAG:           %[[OF_PROD:.*]] = aie.lock(%[[T2]]) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK-DAG:           %[[OF_CONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK-DAG:           %[[IF_B0:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK-DAG:           %[[IF_B1:.*]] = aie.buffer(%[[T2]]) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32>
// CHECK-DAG:           %[[IF_PROD:.*]] = aie.lock(%[[T2]]) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK-DAG:           %[[IF_CONS:.*]] = aie.lock(%[[T2]]) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK-DAG:           %{{.*}} = aie.lock(%[[SHIM]]) {init = 0 : i32, sym_name = "input_fifo_prod_lock_0"}
// CHECK-DAG:           %{{.*}} = aie.lock(%[[SHIM]]) {init = 0 : i32, sym_name = "input_fifo_cons_lock_0"}
// CHECK-DAG:           aie.flow(%[[SHIM]], DMA : 0, %[[T2]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[T2]], DMA : 0, %[[SHIM]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[SHIM]], DMA : 1, %[[T4]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[T4]], DMA : 0, %[[SHIM]], DMA : 1)
// CHECK-DAG:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[SHIM]], MM2S, 0)
// CHECK-DAG:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[SHIM]], S2MM, 0)
// CHECK-DAG:           aie.shim_dma_allocation @input_fifo2_shim_alloc(%[[SHIM]], MM2S, 1)
// CHECK-DAG:           aie.shim_dma_allocation @output_fifo2_shim_alloc(%[[SHIM]], S2MM, 1)
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK-DAG:             %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:             %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[I0:.*]] = arith.constant 0 : i32
// CHECK-DAG:             %[[I1:.*]] = arith.constant 1 : i32
// CHECK-DAG:             %[[I2:.*]] = arith.constant 2 : i32
// CHECK:             %{{.*}}:2 = scf.for %{{.*}} = %[[C0]] to %[[C10]] step %[[C1]] iter_args(%[[OIDX:.*]] = %[[I0]], %[[IIDX:.*]] = %[[I0]]) -> (i32, i32) {
// CHECK:               aie.use_lock(%[[OF_PROD]], AcquireGreaterEqual, %[[I1]])
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
// CHECK:               aie.use_lock(%[[IF_CONS]], AcquireGreaterEqual, %[[I1]])
// CHECK:               %[[IC:.*]] = arith.index_cast %[[IIDX]] : i32 to index
// CHECK:               %[[IB:.*]] = scf.index_switch %[[IC]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[IF_B1]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[IF_B0]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @passthrough_10_i32(%[[IB]], %[[OB]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(%[[IF_PROD]], Release, %[[I1]])
// CHECK:               %[[IN:.*]] = arith.addi %[[IIDX]], %[[I1]] : i32
// CHECK:               %[[ICMP:.*]] = arith.cmpi sge, %[[IN]], %[[I2]] : i32
// CHECK:               %[[ISEL:.*]] = arith.select %[[ICMP]], %[[I0]], %[[IN]] : i32
// CHECK:               aie.use_lock(%[[OF_CONS]], Release, %[[I1]])
// CHECK:               %[[ON:.*]] = arith.addi %[[OIDX]], %[[I1]] : i32
// CHECK:               %[[OCMP:.*]] = arith.cmpi sge, %[[ON]], %[[I2]] : i32
// CHECK:               %[[OSEL:.*]] = arith.select %[[OCMP]], %[[I0]], %[[ON]] : i32
// CHECK:               scf.yield %[[OSEL]], %[[ISEL]] : i32, i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           } {dynamic_objfifo_lowering = true}
// CHECK:           %{{.*}} = aie.core(%[[T4]]) {
// CHECK-DAG:             %[[S2:.*]] = arith.constant 2 : index
// CHECK-DAG:             %[[S10:.*]] = arith.constant 10 : index
// CHECK-DAG:             %[[S0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[SI1:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %{{.*}} = %[[S0]] to %[[S10]] step %[[S2]] {
// CHECK:               aie.use_lock(%[[OF2_PROD]], AcquireGreaterEqual, %[[SI1]])
// CHECK:               aie.use_lock(%[[IF2_CONS]], AcquireGreaterEqual, %[[SI1]])
// CHECK:               func.call @passthrough_10_i32(%[[IF2_B0]], %[[OF2_B0]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(%[[IF2_PROD]], Release, %[[SI1]])
// CHECK:               aie.use_lock(%[[OF2_CONS]], Release, %[[SI1]])
// CHECK:               aie.use_lock(%[[OF2_PROD]], AcquireGreaterEqual, %[[SI1]])
// CHECK:               aie.use_lock(%[[IF2_CONS]], AcquireGreaterEqual, %[[SI1]])
// CHECK:               func.call @passthrough_10_i32(%[[IF2_B1]], %[[OF2_B1]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               aie.use_lock(%[[IF2_PROD]], Release, %[[SI1]])
// CHECK:               aie.use_lock(%[[OF2_CONS]], Release, %[[SI1]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T2]]) {
// CHECK:             %[[M1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[IF_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[IF_PROD]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[IF_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF_CONS]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF_CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[OF_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF_PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF_CONS]], AcquireGreaterEqual, %[[M1]])
// CHECK:             aie.dma_bd(%[[OF_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF_PROD]], Release, %[[M1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %{{.*}} = aie.mem(%[[T4]]) {
// CHECK:             %[[N1:.*]] = arith.constant 1 : i32
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[IF2_PROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[IF2_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF2_CONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[IF2_PROD]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[IF2_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[IF2_CONS]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF2_CONS]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[OF2_B0]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF2_PROD]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF2_CONS]], AcquireGreaterEqual, %[[N1]])
// CHECK:             aie.dma_bd(%[[OF2_B1]] : memref<10xi32> offset = 0 len = 10)
// CHECK:             aie.use_lock(%[[OF2_PROD]], Release, %[[N1]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }

module {
  aie.device(npu1_1col) {
    func.func @passthrough_10_i32(%line_in: memref<10xi32>, %line_out: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    aie.objectfifo @input_fifo2(%tile_0_0, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo2(%tile_0_4, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %1 = aie.objectfifo.acquire @output_fifo (Produce, 1) : memref<10xi32>
        %3 = aie.objectfifo.acquire @input_fifo (Consume, 1) : memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo (Consume, 1)
        aie.objectfifo.release @output_fifo (Produce, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %1 = aie.objectfifo.acquire @output_fifo2 (Produce, 1) : memref<10xi32>
        %3 = aie.objectfifo.acquire @input_fifo2 (Consume, 1) : memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo2 (Consume, 1)
        aie.objectfifo.release @output_fifo2 (Produce, 1)
      }

      aie.end
    }
  }
}
