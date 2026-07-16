//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 

// This test ensures that the proper error is emitted if the user attempts to sepcify more than
// the architecturally possible number of data layout transformation dimensions in a `aie.dma_bd`
// operation inside the runtime sequence. The DMAConfigureTaskOp verifier now rejects this at
// verification time (a shim BD supports 3 access dimensions plus the hoisted iteration
// dimension), before the aie-dma-tasks-to-npu lowering guard would fire.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+1 {{Cannot give more than 4 dimensions}}
          aie.dma_bd(%arg0 : memref<32xi8> offset = 4 len = 32 sizes = [1, 1, 2, 2, 4] strides = [4, 4, 4, 8, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

