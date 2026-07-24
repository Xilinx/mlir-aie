//===- pipeline.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// End-to-end through the real NPU lowering order: from a plain objectFIFO +
// dma_channel_reset_for, the binding must survive aie-materialize-runtime-
// sequences (which whitelists the symbols a runtime sequence may reference) and
// the op must lower to the reset + set_lock + START_QUEUE trio. This is the path
// a lit test that runs the lowering pass in isolation does NOT exercise.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-bd-ids \
// RUN:   --aie-materialize-runtime-sequences --aie-lower-dma-channel-reset-for \
// RUN:   --aie-dma-to-npu --aie-lower-set-lock --aie-lower-dma-channel-reset \
// RUN:   %s | FileCheck %s

// Consumer core tile (0,2), S2MM channel 0. reset_for emits the reset pulse, the
// lock re-arms, and an aiex.npu.push_queue; aie-dma-to-npu (run right after, per
// the real pipeline order) lowers the push_queue to the START_QUEUE write. Only
// the two CORE consumer locks are re-armed; the shim producer's are the host's.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.runtime_sequence
// CHECK: aiex.npu.maskwrite32(%{{.*}}, %{{.*}}, %{{.*}}) {column = 0 : i32, row = 2 : i32}
// CHECK: aiex.npu.maskwrite32(%{{.*}}, %{{.*}}, %{{.*}}) {column = 0 : i32, row = 2 : i32}
// CHECK: aiex.npu.write32(%{{.*}}, %{{.*}}) {column = 0 : i32, row = 2 : i32}
// CHECK: aiex.npu.write32(%{{.*}}, %{{.*}}) {column = 0 : i32, row = 2 : i32}
// The START_QUEUE write comes from the lowered push_queue (absolute address, no
// column/row attrs).
// CHECK: aiex.npu.write32(%{{.*}}, %{{.*}}) : i32, i32
// No high-level re-arm op, no push_queue, and no binding survive to the end.
// CHECK-NOT: aiex.dma_channel_reset_for
// CHECK-NOT: aiex.dma_channel_reset(
// CHECK-NOT: aiex.set_lock
// CHECK-NOT: aiex.npu.push_queue
// CHECK-NOT: aie.objectfifo_rearm_binding
module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of (%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@of)
    }
  }
}
