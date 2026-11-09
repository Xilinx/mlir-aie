// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --verify-diagnostics %s -o /dev/null

module {
  aie.device(npu2) {
    %core_tile = aie.tile(0, 2)
    %shim = aie.tile(0, 0)
    aie.objectfifo @out(%core_tile, {%shim}, 2 : i32) {release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%core_tile) {
      %obj = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }
    aie.runtime_sequence() {
      // expected-error @+1 {{does not support core-pushed DMA channels}}
      aiex.dma_channel_reset_for(@out)
    }
  }
}
