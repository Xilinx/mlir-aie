//===- objectfifo_lock_buffer_aie2.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify that aie.objectfifo.lock and aie.objectfifo.buffer ops are correctly
// lowered by the stateful transform on AIE2. The lock op should resolve to
// the producer/consumer lock SSA values based on port direction.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @test_objectfifo_lock {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[TILE0:.*]] = aie.tile(1, 2)
// CHECK:     %[[TILE1:.*]] = aie.tile(1, 3)
// CHECK:     %[[BUF0:.*]] = aie.buffer(%[[TILE0]]) {sym_name = "of0_buff_0"} : memref<256xi32>
// CHECK:     %[[BUF1:.*]] = aie.buffer(%[[TILE0]]) {sym_name = "of0_buff_1"} : memref<256xi32>
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%[[TILE0]], 0) {init = 2 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%[[TILE0]], 1) {init = 0 : i32, sym_name = "of0_cons_lock_0"}

// For Produce port on AIE2:
//   acq_lock = prod_lock (lock[0])
//   rel_lock = cons_lock (lock[1])
// CHECK:     %[[CORE:.*]] = aie.core(%[[TILE0]]) {
// CHECK:       func.call @kernel(%[[BUF0]], %[[PROD_LOCK]], %[[CONS_LOCK]])
// CHECK:       aie.end

module @test_objectfifo_lock {
 aie.device(xcve2302) {
  %tile12 = aie.tile(1, 2)
  %tile13 = aie.tile(1, 3)

  aie.objectfifo @of0(%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

  func.func private @kernel(%buf: memref<256xi32>,
                             %acq_lock: index, %rel_lock: index) -> ()

  %core12 = aie.core(%tile12) {
    %buf = aie.objectfifo.buffer @of0 (0) : memref<256xi32>
    %acq_lock, %rel_lock = aie.objectfifo.lock @of0 (Produce) : (index, index)
    func.call @kernel(%buf, %acq_lock, %rel_lock)
      : (memref<256xi32>, index, index) -> ()
    aie.end
  } { link_with = "kernel.o" }
 }
}
