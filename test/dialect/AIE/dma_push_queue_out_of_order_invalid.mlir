// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %core = aie.core(%tile) {
      // expected-error@+1 {{out_of_order is only supported on S2MM channels}}
      aie.dma_push_queue(MM2S : 0) bd_id 3 {out_of_order}
      aie.end
    }
  }
}
