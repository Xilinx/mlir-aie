// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-generate-column-control-overlay='configuration=true adaptive=false' %s | FileCheck %s
// RUN: aie-opt --aie-generate-column-control-overlay='configuration=true adaptive=false cover-columns=1' %s | FileCheck %s --check-prefix=EXTENDED

// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK-NOT: ctrlpkt_col1
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CHECK-NOT: ctrlpkt_col1
// CHECK: aie.shim_dma_allocation @ctrlpkt_col2_mm2s_chan0
// CHECK-NOT: ctrlpkt_col1
// EXTENDED-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// EXTENDED-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// EXTENDED-DAG: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0
// EXTENDED-DAG: aie.shim_dma_allocation @ctrlpkt_col2_mm2s_chan0

module {
  aie.device(npu2) @application {
    %left = aie.tile(0, 2)
    %right = aie.tile(2, 2)
  }
}
