// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: split-file %s %t
// RUN: not aie-translate --aie-generate-xaie %t/attribute.mlir 2>&1 | FileCheck %s
// RUN: not aie-translate --aie-generate-xaie %t/push.mlir 2>&1 | FileCheck %s
// RUN: not aie-translate --aie-generate-xaie %t/poll.mlir 2>&1 | FileCheck %s
// CHECK: core-issued configuration is not supported by the XAIEV2 backend

//--- attribute.mlir
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %core = aie.core(%tile) {
      aie.end
    } {processor_bus}
  }
}
//--- push.mlir
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %core = aie.core(%tile) {
      aie.dma_push_queue(S2MM : 0) bd_id 0 {out_of_order}
      aie.end
    }
  }
}
//--- poll.mlir
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %core = aie.core(%tile) {
      aie.dma_poll_for_queue_space(S2MM : 0)
      aie.end
    }
  }
}
