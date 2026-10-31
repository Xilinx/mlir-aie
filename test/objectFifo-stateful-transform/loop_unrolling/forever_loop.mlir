//===- forever_loop.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// A core that runs forever, written as an scf.while whose condition is always
// true, is unrolled like an scf.for: its body is repeated once per rotation of
// the objectFifos it accesses (here lcm(2, 3) = 6 times), so every buffer and
// lock is a constant. Each copy picks up where the previous one left off, so
// @of_out, which advances two elements per iteration, cycles 0, 2, 1. An
// scf.while that can exit is left rolled.

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile2 = aie.tile(0, 2)
    %tile3 = aie.tile(0, 3)

    aie.objectfifo @of_in(%shim, {%tile2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile2, {%shim}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_until(%shim, {%tile3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    func.func private @work(memref<16xi32>, memref<16xi32>)
    func.func private @more() -> i1
    func.func private @use(memref<16xi32>)

    // CHECK-LABEL: aie.core(%{{.*}}tile_0_2)
    // CHECK:         scf.while : () -> () {
    // CHECK:           scf.condition(%{{.*}})
    // CHECK:         } do {
    // CHECK:           func.call @work(%of_in__cons__buf0, %of_out__buf0)
    // CHECK:           func.call @work(%of_in__cons__buf1, %of_out__buf2)
    // CHECK:           func.call @work(%of_in__cons__buf0, %of_out__buf1)
    // CHECK:           func.call @work(%of_in__cons__buf1, %of_out__buf0)
    // CHECK:           func.call @work(%of_in__cons__buf0, %of_out__buf2)
    // CHECK:           func.call @work(%of_in__cons__buf1, %of_out__buf1)
    // CHECK-NOT:       func.call
    // CHECK:           scf.yield
    // CHECK-NOT:     aie.unroll_hint
    // CHECK:         aie.end
    aie.core(%tile2) {
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %in = aie.objectfifo.acquire @of_in(Consume, 1) : memref<16xi32>
        %out0, %out1 = aie.objectfifo.acquire @of_out(Produce, 2) : memref<16xi32>, memref<16xi32>
        func.call @work(%in, %out0) : (memref<16xi32>, memref<16xi32>) -> ()
        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 2)
        scf.yield
      }
      aie.end
    }

    // CHECK-LABEL: aie.core(%{{.*}}tile_0_3)
    // CHECK:         %[[SLOT:.*]] = memref.alloca() : memref<memref<16xi32>>
    // CHECK:         scf.while
    // CHECK:           %[[MORE:.*]] = func.call @more()
    // CHECK:           scf.condition(%[[MORE]])
    // CHECK:         } do {
    // CHECK:           %[[BUF:.*]] = memref.load %[[SLOT]][] : memref<memref<16xi32>>
    // CHECK:           func.call @use(%[[BUF]])
    // CHECK-NOT:       func.call
    // CHECK:           scf.yield
    // CHECK-NOT:     aie.unroll_hint
    // CHECK:         aie.end
    aie.core(%tile3) {
      scf.while : () -> () {
        %more = func.call @more() : () -> i1
        scf.condition(%more)
      } do {
        %until = aie.objectfifo.acquire @of_until(Consume, 1) : memref<16xi32>
        func.call @use(%until) : (memref<16xi32>) -> ()
        aie.objectfifo.release @of_until(Consume, 1)
        scf.yield
      }
      aie.end
    }
  }
}
