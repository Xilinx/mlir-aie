//===- objectfifo_lock_buffer_depth3_aie2.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify that aie.objectfifo.lock and aie.objectfifo.buffer ops work correctly
// with depth-3 ObjectFIFOs (triple buffering).

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @test_objectfifo_lock_depth3 {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[TILE0:.*]] = aie.tile(1, 2)
// CHECK:     %[[TILE1:.*]] = aie.tile(1, 3)
// CHECK:     %[[BUF0:.*]] = aie.buffer(%[[TILE0]]) {sym_name = "of0_buff_0"} : memref<256xi32>
// CHECK:     %[[BUF1:.*]] = aie.buffer(%[[TILE0]]) {sym_name = "of0_buff_1"} : memref<256xi32>
// CHECK:     %[[BUF2:.*]] = aie.buffer(%[[TILE0]]) {sym_name = "of0_buff_2"} : memref<256xi32>
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%[[TILE0]], 0) {init = 3 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%[[TILE0]], 1) {init = 0 : i32, sym_name = "of0_cons_lock_0"}

// For Produce port on AIE2 with depth 3:
//   acq_lock = prod_lock, rel_lock = cons_lock
//   Three buffers: buf_0, buf_1, buf_2
// CHECK:     %[[CORE:.*]] = aie.core(%[[TILE0]]) {
// CHECK:       func.call @kernel(%[[BUF0]], %[[BUF1]], %[[BUF2]], %[[PROD_LOCK]], %[[CONS_LOCK]])
// CHECK:       aie.end

// Consume port on the consumer tile:
//   acq_lock = cons_lock, rel_lock = prod_lock
// CHECK:     %[[CORE1:.*]] = aie.core(%[[TILE1]]) {
// CHECK:       func.call @consumer(%[[BUF0]], %[[BUF1]], %[[BUF2]], %[[CONS_LOCK]], %[[PROD_LOCK]])
// CHECK:       aie.end

module @test_objectfifo_lock_depth3 {
 aie.device(xcve2302) {
  %tile12 = aie.tile(1, 2)
  %tile13 = aie.tile(1, 3)

  aie.objectfifo @of0(%tile12, {%tile13}, 3 : i32) : !aie.objectfifo<memref<256xi32>>

  func.func private @kernel(%buf0: memref<256xi32>, %buf1: memref<256xi32>,
                             %buf2: memref<256xi32>,
                             %acq_lock: index, %rel_lock: index) -> ()
  func.func private @consumer(%buf0: memref<256xi32>, %buf1: memref<256xi32>,
                               %buf2: memref<256xi32>,
                               %acq_lock: index, %rel_lock: index) -> ()

  %core12 = aie.core(%tile12) {
    %buf0 = aie.objectfifo.buffer @of0 (0) : memref<256xi32>
    %buf1 = aie.objectfifo.buffer @of0 (1) : memref<256xi32>
    %buf2 = aie.objectfifo.buffer @of0 (2) : memref<256xi32>
    %acq_lock, %rel_lock = aie.objectfifo.lock @of0 (Produce) : (index, index)
    func.call @kernel(%buf0, %buf1, %buf2, %acq_lock, %rel_lock)
      : (memref<256xi32>, memref<256xi32>, memref<256xi32>, index, index) -> ()
    aie.end
  } { link_with = "kernel.o" }

  %core13 = aie.core(%tile13) {
    %buf0 = aie.objectfifo.buffer @of0 (0) : memref<256xi32>
    %buf1 = aie.objectfifo.buffer @of0 (1) : memref<256xi32>
    %buf2 = aie.objectfifo.buffer @of0 (2) : memref<256xi32>
    %acq_lock, %rel_lock = aie.objectfifo.lock @of0 (Consume) : (index, index)
    func.call @consumer(%buf0, %buf1, %buf2, %acq_lock, %rel_lock)
      : (memref<256xi32>, memref<256xi32>, memref<256xi32>, index, index) -> ()
    aie.end
  } { link_with = "consumer.o" }
 }
}
