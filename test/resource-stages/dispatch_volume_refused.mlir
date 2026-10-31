//===- dispatch_volume_refused.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc's stages verify a parameter-waiting Worker's fifo volume per dispatch
// before the scratchpad lowering replaces the waits and syncs that check reads,
// so a mismatch is refused through the whole pipeline, not only by
// aie-objectfifo-verify on its own.

// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages)' %s

// One object handed over per dispatch, two drained by the host.
module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.objectfifo @out(%tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    %core = aie.core(%tile) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        // expected-error@+1 {{hands over 1 objects through @out_prod for each dispatch, one pass of a loop that waits for each dispatch's parameters, but each run of runtime sequence @sequence moves 2, so the core drifts out of step with the dispatches}}
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<2xi32>
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    // expected-note@+1 {{the runtime sequence is here}}
    aie.runtime_sequence @sequence(%arg0: memref<4xi32>) {
      aiex.sync_scratchpad_parameters_from_host
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4 sizes = [1, 1, 1, 4] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
