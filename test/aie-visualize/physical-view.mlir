//===- physical-view.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --physical --split-packet-ports --device=npu1_device %s | FileCheck %s --check-prefix=NPU1 --implicit-check-not=physical_port_0_2_5_4_source --implicit-check-not=switchbox_4_0
// RUN: aie-visualize --physical --device=npu2_device %s | FileCheck %s --check-prefix=NPU2 --implicit-check-not=switchbox_8_0
// RUN: not aie-visualize --physical --contract=port %s 2>&1 | FileCheck %s --check-prefix=CONTRACT
// RUN: not aie-visualize --physical --emit-dot %s 2>&1 | FileCheck %s --check-prefix=OUTPUT-MODES
// RUN: not aie-visualize --physical --emit-json %s 2>&1 | FileCheck %s --check-prefix=OUTPUT-MODES
// RUN: not aie-visualize --physical --scale=0.25 %s 2>&1 | FileCheck %s --check-prefix=PHYSICAL-OPTIONS
// RUN: not aie-visualize --physical --contract-balanced-fanout %s 2>&1 | FileCheck %s --check-prefix=PHYSICAL-OPTIONS

// The model supplies every switchbox and port, including unused resources.
// NPU1-DAG: switchbox_0_2 [shape=box{{.*}}pos="0.000000,1008.000000!"
// NPU1-DAG: physical_port_0_2_5_3_source [shape=point{{.*}}color="#d9dde1"{{.*}}class="physical-port unused"
// NPU1-DAG: physical_port_0_2_5_5_dest [shape=point

// Fixed positions depend only on the tile, bundle, channel, and direction.
// NPU1-DAG: physical_port_0_2_0_0_source {{.*}} pos="-104.400000,1112.400000!"
// NPU1-DAG: physical_port_0_2_5_0_dest {{.*}} pos="-61.200000,1139.040000!"

// The used crosspoint is dark and wider. Other legal crosspoints stay faint.
// NPU1-DAG: physical_port_0_2_0_0_source -> physical_port_0_2_5_0_dest [color="#c51b32", penwidth=1.6, class="physical-track used"]
// NPU1-DAG: physical_port_0_2_0_0_source -> physical_port_0_2_5_1_dest [color="#e7eaed", penwidth=0.18, class="physical-track unused"]
// NPU1-DAG: physical_port_0_2_5_0_dest -> physical_port_0_3_3_0_source [color="#c51b32", penwidth=1.6, class="physical-intertile-track used"]
// NPU1-DAG: physical_port_0_2_6_0_dest -> physical_port_1_2_4_0_source [color="#e1e5e8", penwidth=0.22, class="physical-intertile-track unused"]

// Packet aliases occupy the physical port coordinates rather than separate lanes.
// NPU1-DAG: semantic_p_0_2_0_0_k3 {{.*}}pos="-104.400000,1112.400000!"
// NPU1-DAG: semantic_p_0_2_0_0_k4 {{.*}}pos="-104.400000,1112.400000!"

// DMA channels and all model-reported BDs occupy a box beside the tile.
// NPU1-DAG: dma_box_0_1 {{.*}}label=""
// NPU1-DAG: dma_title_0_1 {{.*}}label="DMA resources"
// NPU1-DAG: dma_s2mm_0_1_0 {{.*}}class="physical-dma-channel used"
// NPU1-DAG: bd_0_1_7 {{.*}}color="#c51b32"{{.*}}class="physical-bd used"
// NPU1-DAG: bd_0_1_47 [shape=box
// NPU1-DAG: bd_group_0_1_0_24 {{.*}}label="channels 0, 2, 4: BD0-23"
// NPU1-DAG: bd_group_0_1_24_48 {{.*}}label="channels 1, 3, 5: BD24-47"

// Full NPU1 and NPU2 fixtures expose different target-model array extents.
// NPU1-DAG: switchbox_3_5 [shape=box
// NPU2-DAG: switchbox_7_5 [shape=box

// CONTRACT: --physical does not support --contract
// OUTPUT-MODES: --emit-dot, --emit-json, and --physical are mutually exclusive
// PHYSICAL-OPTIONS: --physical does not support topology layout and styling options

module {
  aie.device(npu1) @npu1_device {
    %source = aie.tile(0, 2)
    %dest = aie.tile(0, 3)
    %mem = aie.tile(0, 1)
    %buffer = aie.buffer(%mem) : memref<16xi32>

    aie.memtile_dma(%mem) {
      %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16) {bd_id = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }

    aie.packet_flow(3, mask = 31) {
      aie.packet_source<%source, Core : 0>
      aie.packet_dest<%dest, Core : 0>
    } via (%source : Core : 0 -> North : 0,
           %dest : South : 0 -> Core : 0)
    aie.packet_flow(4, mask = 31) {
      aie.packet_source<%source, Core : 0>
      aie.packet_dest<%dest, Core : 0>
    } via (%source : Core : 0 -> North : 0,
           %dest : South : 0 -> Core : 0)
    aie.packet_flow(5, mask = 31) {
      aie.packet_source<%source, Core : 0>
      aie.packet_dest<%mem, DMA : 0>
    } via (%source : Core : 0 -> South : 1,
           %mem : North : 1 -> DMA : 0)
  }

  aie.device(npu2) @npu2_device {
    %source = aie.tile(0, 2)
    %dest = aie.tile(0, 3)
    aie.flow(%source, Core : 0, %dest, Core : 0)
      via (%source : Core : 0 -> North : 0,
           %dest : South : 0 -> Core : 0)
  }
}
