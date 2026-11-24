//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --pass-pipeline='builtin.module(aie-materialize-runtime-sequences,aie.device(aie-assign-runtime-sequence-bd-ids{warn-unsafe-bd-reuse=true}))' \
// RUN:   --mlir-disable-threading %s -o /dev/null 2> %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=COUNT < %t

// Materialization inlines @seq at each of its three calls, so in @main its
// unsafe reuse happens three times from one source transfer. Each device
// reports it once, @main with how many more times it happened.

// COUNT-COUNT-2: warning: reuses buffer descriptor ID 0 on tile (0,0)
// COUNT-NOT: warning:
// CHECK: note: and 2 more times from this transfer, on tile (0,0)

module {
  aie.device(npu2) @copy {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence @seq(%arg0: memref<512xi32>) {
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_free_task(%t0)
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 256 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
    }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg0: memref<512xi32>) {
      aiex.configure @copy {
        aiex.run @seq(%arg0) : (memref<512xi32>)
        aiex.run @seq(%arg0) : (memref<512xi32>)
        aiex.run @seq(%arg0) : (memref<512xi32>)
      }
    }
  }
}
