// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie-generate-column-control-overlay | FileCheck %s --check-prefix=TOKENS
// RUN: aie-opt %s --aie-generate-column-control-overlay='configuration=true adaptive=false' | FileCheck %s --check-prefix=CONFIG
// RUN: aie-opt %s --aie-generate-column-control-overlay --aie-import-overlay --aie-create-pathfinder-flows | FileCheck %s --check-prefix=IMPORT

// TOKENS-LABEL: aie.device(npu2) @application
// TOKENS-NOT: aie.packet_flow
// TOKENS: overlay = @ctrl_pkt_overlay
// TOKENS-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// TOKENS-COUNT-2: aie.packet_flow(15)
// TOKENS-NOT: aie.packet_flow
// TOKENS-NOT: aie.shim_dma_allocation

// CONFIG-LABEL: aie.device(npu2) @application
// CONFIG-NOT: aie.switchbox
// CONFIG: overlay = @ctrl_pkt_overlay
// CONFIG-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CONFIG-NOT: aie.packet_flow
// CONFIG-DAG: aie.tile(1, 5)
// CONFIG-DAG: aie.tile(0, 5)
// CONFIG-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CONFIG-DAG: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0
// CONFIG-DAG: aie.masterset(TileControl : 0, {{.*}}) {aie.is_ctrl_overlay{{.*}}}
// CONFIG-NOT: ctrlpkt_col2

// IMPORT-LABEL: aie.device(npu2) @application
// IMPORT-NOT: aie.packet_flow
// IMPORT: aie.switchbox
// IMPORT: aie.packet_rules(TileControl : 0)
// IMPORT: overlay = @ctrl_pkt_overlay

module {
  aie.device(npu2) @application {
    %shim0 = aie.tile(0, 0)
    %core0 = aie.tile(0, 2)
    %shim1 = aie.tile(1, 0)
    %core1 = aie.tile(1, 2)
  }
}
