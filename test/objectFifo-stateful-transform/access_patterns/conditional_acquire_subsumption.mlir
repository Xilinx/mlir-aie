//===- conditional_acquire_subsumption.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// An acquire before an scf.if, and a smaller acquire in each arm. An acquire
// names the total the core wants to hold, so the arms ask for no more than the
// core already holds and must take no further locks on either path. The
// condition is read from a buffer so the branch survives to the lowering.

// CHECK-LABEL: @conditionalAcquireSubsumption
// CHECK:         aie.core(%{{.*}}tile_4_3)
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : i32

// One acquire covers the whole region, and neither arm takes another lock.
// CHECK:           aie.use_lock(%{{.*}}fifo_cons_cons_lock_0, AcquireGreaterEqual, %[[C2]])
// CHECK:           scf.if
// CHECK-NOT:         AcquireGreaterEqual
// CHECK:             aie.use_lock(%{{.*}}fifo_cons_prod_lock_0, Release, %[[C1]])
// CHECK:           } else {
// CHECK-NOT:         AcquireGreaterEqual
// CHECK:             aie.use_lock(%{{.*}}fifo_cons_prod_lock_0, Release, %[[C1]])
// CHECK:           }
// CHECK:           aie.use_lock(%{{.*}}fifo_cons_prod_lock_0, Release, %[[C1]])
// CHECK:           aie.end

module @conditionalAcquireSubsumption {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile43 = aie.tile(4, 3)
    %flag = aie.buffer(%tile43) {sym_name = "flag"} : memref<1xi32>

    aie.objectfifo @fifo (%tile12, {%tile43}, 3 : i32) : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %v = arith.constant 11 : i32
      scf.for %i = %c0 to %c3 step %c1 {
        %e = aie.objectfifo.acquire @fifo (Produce, 1) : memref<16xi32>
        memref.store %v, %e[%c0] : memref<16xi32>
        aie.objectfifo.release @fifo (Produce, 1)
      }
      aie.end
    }

    %core43 = aie.core(%tile43) {
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0 : i32
      %fv = memref.load %flag[%c0] : memref<1xi32>
      %cond = arith.cmpi sgt, %fv, %zero : i32
      %e0, %e1 = aie.objectfifo.acquire @fifo (Consume, 2) : memref<16xi32>, memref<16xi32>
      %v0 = memref.load %e0[%c0] : memref<16xi32>
      scf.if %cond {
        %t = aie.objectfifo.acquire @fifo (Consume, 1) : memref<16xi32>
        %v1 = memref.load %t[%c0] : memref<16xi32>
        aie.objectfifo.release @fifo (Consume, 1)
      } else {
        %f = aie.objectfifo.acquire @fifo (Consume, 1) : memref<16xi32>
        %v2 = memref.load %f[%c0] : memref<16xi32>
        aie.objectfifo.release @fifo (Consume, 1)
      }
      aie.objectfifo.release @fifo (Consume, 1)
      aie.end
    }
 }
}
