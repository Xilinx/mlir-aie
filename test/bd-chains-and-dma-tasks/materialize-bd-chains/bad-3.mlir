//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-materialize-bd-chains %s

// This test ensures the proper error gets emitted if the user attempts to
// reference a BD chain that has not been defined.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.runtime_sequence(%buf: memref<8xi16>) {
      // expected-error@+1 {{symbol does not reference valid BD chain}}
      %t1 = aiex.dma_start_bd_chain @concat(%buf) : (memref<8xi16>) 
                                    on (%tile_0_0, MM2S, 0) 
      aiex.dma_await_task(%t1)
    }

  }
}

