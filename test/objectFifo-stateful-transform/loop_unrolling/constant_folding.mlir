//===- constant_folding.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test checks that, once dynamic objectFifos are disabled, the loop is
// unrolled by the objectFifo rotation period and the *dynamic runtime
// bookkeeping* the stateful transform emits is folded into static constants.
//
// Under dynamic lowering the consumer acquire lowers to a runtime-computed lock
// amount:
//     %held  = <load runtime held counter>
//     %delta = arith.maxsi(arith.subi(%acqNum, %held), 0)
//     aie.use_lock(%cons_lock, AcquireGreaterEqual, %delta)
// and the acquired buffer is selected at runtime with an scf.index_switch.
//
// After unrolling by the buffer depth followed by mem2reg + canonicalize + sccp
// + canonicalize, the modular held counter becomes loop-invariant, so:
//   * the AcquireGreaterEqual amount folds to a constant (arith.constant 1),
//   * the arith.subi / arith.maxsi bookkeeping disappears, and
//   * the scf.index_switch collapses to a constant buffer per unrolled body.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" --aie-assign-lock-ids --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[CONS_BUFF0:.*]] = aie.buffer(%{{.*}}) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[CONS_BUFF1:.*]] = aie.buffer(%{{.*}}) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}) {init = 2 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           aie.core
// The runtime-computed acquire amount folds to a single constant...
// CHECK:             %[[C1:.*]] = arith.constant 1 : i32
// ...and the loop is unrolled by the depth (2), each body touching a constant
// buffer with the constant, folded AcquireGreaterEqual amount -- no runtime
// index selection and no acquire-count arithmetic remain.
// CHECK:             scf.for
// CHECK:               aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, %[[C1]])
// CHECK:               func.call @work(%[[CONS_BUFF0]])
// CHECK:               aie.use_lock(%[[PROD_LOCK]], Release, %[[C1]])
// CHECK:               aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, %[[C1]])
// CHECK:               func.call @work(%[[CONS_BUFF1]])
// CHECK:               aie.use_lock(%[[PROD_LOCK]], Release, %[[C1]])
// CHECK-NOT:           scf.index_switch
// CHECK-NOT:           arith.subi
// CHECK-NOT:           arith.maxsi

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>
    func.func @work(%b: memref<8xi8>) -> () {
      return
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %a = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
        %e = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
        func.call @work(%e) : (memref<8xi8>) -> ()
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.end
    }
  }
}
