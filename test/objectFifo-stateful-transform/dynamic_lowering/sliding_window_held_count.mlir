//===- sliding_window_held_count.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two cores run the same sliding window over a depth-4 objectFifo: every
// iteration acquires 2 elements and releases 1. They differ only in whether the
// core acquires an element before entering the loop, and that decides whether
// the lock values are constants or computed at run time.
//
// core_0_2 enters the loop holding nothing. Its first iteration therefore has to
// acquire 2 elements, while every later iteration only needs 1 more. So each
// iteration acquires a runtime-computed `acq = max(2 - h, 0)`, where `h` counts
// the elements the core already holds.
//
// core_0_3 acquires one element before the loop and never releases it, so it
// enters the loop already holding 1 -- exactly what it holds at the end of every
// iteration. `h` never changes, so each iteration acquires the constant 1.
//
// Unrolling alone does not change this, because it does not change how many
// elements the core holds when it enters the loop. The second run below unrolls
// both loops, which also peels core_0_2's first iteration so that the rest of
// its loop starts holding 1 and its acquires become constant too.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s
// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s --check-prefix=UNROLL

// CHECK-LABEL:   aie.device(npu2) {
// CHECK-DAG:       %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:       %[[T3:.*]] = aie.tile(0, 3)
// CHECK-DAG:       %[[BL_PROD:.*]] = aie.lock(%[[T3]], 0) {init = 3 : i32, sym_name = "acquire_before_loop_cons_prod_lock_0"}
// CHECK-DAG:       %[[BL_CONS:.*]] = aie.lock(%[[T3]], 1) {init = 0 : i32, sym_name = "acquire_before_loop_cons_cons_lock_0"}
// CHECK-DAG:       %[[IN_PROD:.*]] = aie.lock(%[[T2]], 0) {init = 3 : i32, sym_name = "acquire_in_loop_cons_prod_lock_0"}
// CHECK-DAG:       %[[IN_CONS:.*]] = aie.lock(%[[T2]], 1) {init = 0 : i32, sym_name = "acquire_in_loop_cons_cons_lock_0"}

// core_0_2 holds nothing on entry, so it computes how much to acquire.
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK-DAG:         %[[IN_C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:         %[[IN_C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:         %[[IN_C2:.*]] = arith.constant 2 : i32
// CHECK:             %{{.*}}:2 = scf.for %{{.*}} iter_args(%{{.*}} = %[[IN_C0]], %[[HELD:.*]] = %[[IN_C0]]) -> (i32, i32) {
// CHECK:               %[[DELTA:.*]] = arith.subi %[[IN_C2]], %[[HELD]] : i32
// CHECK:               %[[ACQ:.*]] = arith.maxsi %[[DELTA]], %[[IN_C0]] : i32
// CHECK:               aie.use_lock(%[[IN_CONS]], AcquireGreaterEqual, %[[ACQ]])
// CHECK:               %[[HELDACQ:.*]] = arith.addi %[[HELD]], %[[ACQ]] : i32
// CHECK:               aie.use_lock(%[[IN_PROD]], Release, %[[IN_C1]])
// CHECK:               %[[HELDREL:.*]] = arith.subi %[[HELDACQ]], %[[IN_C1]] : i32
// CHECK:               scf.yield %{{.*}}, %[[HELDREL]] : i32, i32
// CHECK:             }
// CHECK:             aie.end

// core_0_3 holds 1 on entry, so every acquire is the constant 1.
// CHECK:           %{{.*}} = aie.core(%[[T3]]) {
// CHECK-DAG:         %[[BL_C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:         %[[BL_C1:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[BL_CONS]], AcquireGreaterEqual, %[[BL_C1]])
// CHECK:             %{{.*}} = scf.for %{{.*}} iter_args(%{{.*}} = %[[BL_C0]]) -> (i32) {
// CHECK-NOT:           arith.maxsi
// CHECK:               aie.use_lock(%[[BL_CONS]], AcquireGreaterEqual, %[[BL_C1]])
// CHECK:               aie.use_lock(%[[BL_PROD]], Release, %[[BL_C1]])
// CHECK-NOT:           arith.maxsi
// CHECK:             }
// CHECK:             aie.end

// UNROLL-DAG:      %[[X_T2:.*]] = aie.tile(0, 2)
// UNROLL-DAG:      %[[X_T3:.*]] = aie.tile(0, 3)

// Unrolled, core_0_2 is peeled so that the remaining loop starts holding 1 and
// every acquire is constant; the peeled iteration acquires 2 up front.
// UNROLL:          %{{.*}} = aie.core(%[[X_T2]]) {
// UNROLL-DAG:        %[[X_C1:.*]] = arith.constant 1 : i32
// UNROLL-DAG:        %[[X_C2:.*]] = arith.constant 2 : i32
// UNROLL:            aie.use_lock(%{{.*}}, AcquireGreaterEqual, %[[X_C2]])
// UNROLL-NOT:        iter_args
// UNROLL-NOT:        arith.maxsi
// UNROLL:            scf.for
// UNROLL:              aie.use_lock(%{{.*}}, AcquireGreaterEqual, %[[X_C1]])
// UNROLL-NOT:          arith.maxsi
// UNROLL:            }
// UNROLL:            aie.end

// Unrolled, core_0_3 carries nothing and every acquire is constant.
// UNROLL:          %{{.*}} = aie.core(%[[X_T3]]) {
// UNROLL-NOT:        iter_args
// UNROLL-NOT:        arith.maxsi
// UNROLL:            aie.end

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @acquire_in_loop(%tile_0_1, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @acquire_before_loop(%tile_0_1, {%tile_0_3}, 4 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %a = aie.objectfifo.acquire @acquire_in_loop(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        %e0 = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        %e1 = aie.objectfifo.subview.access %a[1] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        aie.objectfifo.release @acquire_in_loop(Consume, 1)
      }
      aie.end
    }

    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %p = aie.objectfifo.acquire @acquire_before_loop(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
      %pe = aie.objectfifo.subview.access %p[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %a = aie.objectfifo.acquire @acquire_before_loop(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        %e0 = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        %e1 = aie.objectfifo.subview.access %a[1] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        aie.objectfifo.release @acquire_before_loop(Consume, 1)
      }
      aie.end
    }
  }
}
