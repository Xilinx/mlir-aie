//===- via_DMA_repeat_count_test.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// `via_DMA` and `repeat_count` on one fifo between adjacent tiles. The two
// attributes are independent: via_DMA forces the DMA path that the shared
// memory between neighbours would otherwise replace, and repeat_count then
// decides how many times that chain fires per buffer.

// CHECK-LABEL: @viaDMARepeat
// The producer counts depth * repeat_count; the consumer only counts depth.
// CHECK-DAG:     %[[PROD:.*]] = aie.lock(%{{.*}}tile_1_2) {init = 2 : i32, sym_name = "of_prod_lock_0"}
// CHECK-DAG:     %[[CONS:.*]] = aie.lock(%{{.*}}tile_1_2) {init = 0 : i32, sym_name = "of_cons_lock_0"}
// CHECK-DAG:     %[[CPROD:.*]] = aie.lock(%{{.*}}tile_1_3) {init = 1 : i32, sym_name = "of_cons_prod_lock_0"}
// CHECK-DAG:     %[[CCONS:.*]] = aie.lock(%{{.*}}tile_1_3) {init = 0 : i32, sym_name = "of_cons_cons_lock_0"}

// via_DMA forces a flow between neighbours that would otherwise share memory.
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_1_3, DMA : 0)
// CHECK-DAG:     aie.buffer(%{{.*}}tile_1_3) {sym_name = "of_cons_buff_0"}

// CHECK:         aie.core(%{{.*}}tile_1_2)
// CHECK:           %[[C2:.*]] = arith.constant 2 : i32
// CHECK:           aie.use_lock(%[[PROD]], AcquireGreaterEqual, %[[C2]])
// CHECK:           aie.use_lock(%[[CONS]], Release, %[[C2]])

// CHECK:         aie.core(%{{.*}}tile_1_3)
// CHECK:           %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           aie.use_lock(%[[CCONS]], AcquireGreaterEqual, %[[C1]])
// CHECK:           aie.use_lock(%[[CPROD]], Release, %[[C1]])

module @viaDMARepeat {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of (%tile12, {%tile13}, 1 : i32) {via_DMA = true, repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      %e = aie.objectfifo.acquire @of (Produce, 1) : memref<16xi32>
      memref.store %v, %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of (Produce, 1)
      aie.end
    }

    %core13 = aie.core(%tile13) {
      %c0 = arith.constant 0 : index
      %e = aie.objectfifo.acquire @of (Consume, 1) : memref<16xi32>
      %v = memref.load %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of (Consume, 1)
      aie.end
    }
 }
}
