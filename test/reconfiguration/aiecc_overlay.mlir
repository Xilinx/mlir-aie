//===- aiecc_overlay.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The column control overlay in aiecc.
//
// It runs after objectFIFO lowering, so it takes the shim MM2S channel the data
// leaves free. reconfigure_loadpdi's design, moved to row 3, feeds its core
// through channel 0 and takes control packets through channel 1, under both
// overlay flags.
// The unmoved design (row 2) builds the same way.

// RUN: rm -rf %t && mkdir -p %t/row2 %t/row3 %t/row3_overlay %t/spill
// RUN: sed 's/aie.tile(0, 2)/aie.tile(0, 3)/' %S/../npu-xrt/reconfigure_loadpdi/aie.mlir > %t/row3.mlir
// RUN: cd %t/row2 && %aiecc --load-pdi-to-ctrl-pkt --get=input_physical.mlir %S/../npu-xrt/reconfigure_loadpdi/aie.mlir
// RUN: FileCheck %s --check-prefix=CHANNELS < %t/row2/input_physical.mlir
// RUN: cd %t/row3 && %aiecc --load-pdi-to-ctrl-pkt --get=input_physical.mlir %t/row3.mlir
// RUN: FileCheck %s --check-prefix=CHANNELS < %t/row3/input_physical.mlir
// RUN: cd %t/row3_overlay && %aiecc --generate-ctrl-pkt-overlay --get=input_physical.mlir %t/row3.mlir
// RUN: FileCheck %s --check-prefix=CHANNELS < %t/row3_overlay/input_physical.mlir

// CHANNELS-LABEL: @add_two
// CHANNELS-DAG:   aie.shim_dma_allocation @{{.*}}in{{.*}}(%{{.+}}, MM2S, 0)
// CHANNELS-DAG:   aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1(%{{.+}}, MM2S, 1)
// CHANNELS-NOT:   ctrlpkt_col0_mm2s_chan0

// The design below occupies column 0 only. Its six flows cross from rows 2-3
// to rows 4-5, which column 0's north channels between rows 3 and 4 carry
// alone, but not beside the overlay's channel there, so one detours through
// column 1. aiecc finds the column by routing and covers it.

// RUN: cd %t/spill && %aiecc --load-pdi-to-ctrl-pkt --get=input_physical.mlir %s
// RUN: FileCheck %s --check-prefix=SPILL < %t/spill/input_physical.mlir

// SPILL-LABEL: aie.device(npu2) @cfg
// SPILL-DAG:   aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// SPILL-DAG:   aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0
// SPILL-NOT:   ctrlpkt_col2_
// SPILL-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// SPILL-DAG:   aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// SPILL-DAG:   aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @seq(%a : memref<4xi32>) {
      aiex.configure @cfg {
        aiex.run @cfg_run(%a) : (memref<4xi32>)
      }
    }
  }
  aie.device(npu2) @cfg {
    %s = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    %t4 = aie.tile(0, 4)
    %t5 = aie.tile(0, 5)
    aie.flow(%t2, DMA : 0, %t4, DMA : 0)
    aie.flow(%t2, DMA : 1, %t5, DMA : 0)
    aie.flow(%t3, DMA : 0, %t4, DMA : 1)
    aie.flow(%t3, DMA : 1, %t5, DMA : 1)
    aie.flow(%t2, Core : 0, %t5, Core : 0)
    aie.flow(%t3, Core : 0, %t4, Core : 0)
    aie.runtime_sequence @cfg_run(%a : memref<4xi32>) {
    }
  }
}
