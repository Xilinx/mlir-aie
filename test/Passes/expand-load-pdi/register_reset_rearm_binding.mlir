//===- register_reset_rearm_binding.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt "--aie-expand-load-pdi=register-reset=true" %s | FileCheck %s

// When the outgoing device has a resident objectFIFO (an
// aie.objectfifo_rearm_binding, as left by aie.objectfifo_stateful_transform
// + --aie-assign-bd-ids), register-reset=true re-arms it with
// aiex.dma_channel_reset_for instead of a firmware reset. The binding's tile
// and lock operands are re-declared inside @main under a fresh name, because
// aiex.dma_channel_reset_for resolves its symbol only within its own
// enclosing device and an SSA value cannot cross into the sibling device
// @dev_a where the original binding lives.

module {

    aie.device(npu2) @dev_a {
        %t03 = aie.tile(0, 3)
        %pl = aie.lock(%t03, 0) {init = 1 : i32}
        // CHECK: aie.objectfifo_rearm_binding @of_rearm
        aie.objectfifo_rearm_binding @of_rearm channels(%t03 : index) locks(%pl : index) {
          channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>,
          lock_inits = array<i32: 1>, head_bd_ids = array<i32: 2>,
          repeat_counts = array<i32: 0>
        }
    }

    aie.device(npu2) @dev_b {
        %tile = aie.tile(0, 3)
    }

    aie.device(npu2) @main {
        aie.runtime_sequence (%arg0: memref<1xi32>) {
            aiex.npu.load_pdi { device_ref = @dev_a }
            // CHECK: %[[T:.*]] = aie.tile(0, 3)
            // CHECK: %[[L:.*]] = aie.lock(%[[T]], 0)
            // CHECK: aie.objectfifo_rearm_binding @of_rearm_regreset_0
            // CHECK-SAME: channels(%[[T]] : index) locks(%[[L]] : index)
            // CHECK-SAME: head_bd_ids = array<i32: 2>
            // CHECK-SAME: lock_inits = array<i32: 1>
            // CHECK-SAME: repeat_counts = array<i32: 0>
            // CHECK: aiex.dma_channel_reset_for(@of_rearm_regreset_0)
            aiex.npu.load_pdi { device_ref = @dev_b }
        }
    }

}
