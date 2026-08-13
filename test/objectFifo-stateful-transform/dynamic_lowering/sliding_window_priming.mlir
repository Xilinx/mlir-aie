//===- sliding_window_priming.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Both cores below run the same sliding window (acquire 2, release 1) over a
// depth-4 objectFifo; only the priming differs, and that decides whether the
// acquire count is a constant or a runtime value.
//
// The held counter follows `h -> max(h, acqNumber) - relNumber`, whose fixed
// point here is 1. core_0_2 enters its loop with h = 0, so the first iteration
// acquires 2 and every later one acquires 1: the counter is loop-carried and
// the acquire stays a runtime `max(2 - h, 0)`. core_0_3 acquires one element
// before the loop and never releases it, so the loop is entered at h = 1, which
// is already the fixed point; the counter is loop-invariant, folds away, and
// every lock value becomes constant. Only the periodic buffer index remains as
// an iter_arg.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK-DAG:       %[[T2:.*]] = aie.tile(0, 2)
// CHECK-DAG:       %[[T3:.*]] = aie.tile(0, 3)
// CHECK-DAG:       %[[P_PROD:.*]] = aie.lock(%[[T3]], 0) {init = 3 : i32, sym_name = "primed_cons_prod_lock_0"}
// CHECK-DAG:       %[[P_CONS:.*]] = aie.lock(%[[T3]], 1) {init = 0 : i32, sym_name = "primed_cons_cons_lock_0"}
// CHECK-DAG:       %[[U_PROD:.*]] = aie.lock(%[[T2]], 0) {init = 3 : i32, sym_name = "unprimed_cons_prod_lock_0"}
// CHECK-DAG:       %[[U_CONS:.*]] = aie.lock(%[[T2]], 1) {init = 0 : i32, sym_name = "unprimed_cons_cons_lock_0"}

// Unprimed: the held counter is carried, so the acquire count is computed at run time.
// CHECK:           %{{.*}} = aie.core(%[[T2]]) {
// CHECK-DAG:         %[[U_C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:         %[[U_C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:         %[[U_C2:.*]] = arith.constant 2 : i32
// CHECK:             %{{.*}}:2 = scf.for %{{.*}} iter_args(%{{.*}} = %[[U_C0]], %[[HELD:.*]] = %[[U_C0]]) -> (i32, i32) {
// CHECK:               %[[DELTA:.*]] = arith.subi %[[U_C2]], %[[HELD]] : i32
// CHECK:               %[[ACQ:.*]] = arith.maxsi %[[DELTA]], %[[U_C0]] : i32
// CHECK:               aie.use_lock(%[[U_CONS]], AcquireGreaterEqual, %[[ACQ]])
// CHECK:               %[[HELDACQ:.*]] = arith.addi %[[HELD]], %[[ACQ]] : i32
// CHECK:               aie.use_lock(%[[U_PROD]], Release, %[[U_C1]])
// CHECK:               %[[HELDREL:.*]] = arith.subi %[[HELDACQ]], %[[U_C1]] : i32
// CHECK:               scf.yield %{{.*}}, %[[HELDREL]] : i32, i32
// CHECK:             }
// CHECK:             aie.end

// Primed: entered at the fixed point, so the held counter is gone and the loop
// carries only the buffer index.
// CHECK:           %{{.*}} = aie.core(%[[T3]]) {
// CHECK-DAG:         %[[P_C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:         %[[P_C1:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[P_CONS]], AcquireGreaterEqual, %[[P_C1]])
// CHECK:             %{{.*}} = scf.for %{{.*}} iter_args(%{{.*}} = %[[P_C0]]) -> (i32) {
// CHECK-NOT:           arith.maxsi
// CHECK:               aie.use_lock(%[[P_CONS]], AcquireGreaterEqual, %[[P_C1]])
// CHECK:               aie.use_lock(%[[P_PROD]], Release, %[[P_C1]])
// CHECK-NOT:           arith.maxsi
// CHECK:             }
// CHECK:             aie.end

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @unprimed(%tile_0_1, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @primed(%tile_0_1, {%tile_0_3}, 4 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %a = aie.objectfifo.acquire @unprimed(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        %e0 = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        %e1 = aie.objectfifo.subview.access %a[1] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        aie.objectfifo.release @unprimed(Consume, 1)
      }
      aie.end
    }

    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %p = aie.objectfifo.acquire @primed(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
      %pe = aie.objectfifo.subview.access %p[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %a = aie.objectfifo.acquire @primed(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        %e0 = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        %e1 = aie.objectfifo.subview.access %a[1] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        aie.objectfifo.release @primed(Consume, 1)
      }
      aie.end
    }
  }
}
