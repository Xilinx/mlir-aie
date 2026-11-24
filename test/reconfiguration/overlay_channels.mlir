//===- overlay_channels.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-generate-column-control-overlay="configuration=true " --aie-import-overlay --split-input-file %s | FileCheck %s

// Each column's control packets enter through one shim MM2S channel, the
// lowest one no device of the module moves data through. Data moves through a
// channel that is the source of a circuit flow (column 0) or of a packet flow
// (column 1), or that a shim DMA allocation names (column 2).

// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1(%{{.*}}, MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan1(%{{.*}}, MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col2_mm2s_chan1(%{{.*}}, MM2S, 1)
// CHECK-NOT: mm2s_chan0
module {
  aie.device(npu2) @a {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t10 = aie.tile(1, 0)
    %t12 = aie.tile(1, 2)
    %t20 = aie.tile(2, 0)
    aie.flow(%t00, DMA : 0, %t02, DMA : 0)
    aie.packet_flow(1) {
      aie.packet_source<%t10, DMA : 0>
      aie.packet_dest<%t12, DMA : 0>
    }
    aie.shim_dma_allocation @in2(%t20, MM2S, 0)
  }
}

// -----

// The choice is the same in every device and in the standalone overlay. @b
// moves no data through column 0, but @a moves data through its channel 0.

// CHECK-LABEL: aie.device(npu2) @a
// CHECK: overlay = @ctrl_pkt_overlay
// CHECK-LABEL: aie.device(npu2) @b
// CHECK: overlay = @ctrl_pkt_overlay
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1(%{{.*}}, MM2S, 1)
module {
  aie.device(npu2) @a {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.flow(%t00, DMA : 0, %t02, DMA : 0)
  }
  aie.device(npu2) @b {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
  }
}

// -----

// Covered columns reach every row, without extending into unused columns.
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK-DAG: aie.tile(0, 5)
// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CHECK-DAG: aie.masterset(TileControl : 0, {{.*}}) {aie.is_ctrl_pkt_overlay{{.*}}}
// CHECK-NOT: ctrlpkt_col1
module {
  aie.device(npu2) @a {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
  }
}
