//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-flows-to-dot %s | FileCheck %s

// CHECK: digraph AIE_Routing {
// CHECK-DAG: tile_0_0_bg [label="(0,0)"
// CHECK-DAG: tile_0_1_bg [label="(0,1)"
// CHECK-DAG: tile_0_2_bg [label="(0,2)"

// CHECK-DAG: tile_0_0_sb
// CHECK-DAG: tile_0_1_sb
// CHECK-DAG: tile_0_2_sb

// CHECK-DAG: tile_0_0_dma_mm2s_0 [label="M\nM\n2\nS\n0"
// CHECK-DAG: tile_0_0_dma_s2mm_1 [label="S\n2\nM\nM\n1"
// CHECK-DAG: tile_0_1_dma_s2mm_0 [label="S\n2\nM\nM\n0"
// CHECK-DAG: tile_0_2_dma_mm2s_1 [label="M\nM\n2\nS\n1"
// CHECK-DAG: tile_0_2_core [label="Core"
// CHECK-DAG: tile_0_2_memory [label="Memory"

// Circuit flow (route0): tile_0_0 DMA:0 -> tile_0_1 DMA:0
// CHECK-DAG: tile_0_0_dma_mm2s_0_top -> {{.*}} [label="route0"
// CHECK-DAG: tile_0_1_dma_s2mm_0_top [{{.*}}shape=point

// Packet flow (route1): tile_0_2 DMA:1 -> tile_0_0 DMA:1
// CHECK-DAG: tile_0_2_dma_mm2s_1_top -> {{.*}} [label="route1"
// CHECK-DAG: tile_0_0_dma_s2mm_1_top [{{.*}}shape=point

module @aie_module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 3, %0)
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 1, %0)
      }
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, DMA : 1>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<South : 3, North : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 3, %0)
      aie.packet_rules(North : 3) {
        aie.rule(31, 1, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<North : 3, DMA : 1>
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      aie.connect<South : 0, DMA : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 3, %0)
      aie.packet_rules(North : 3) {
        aie.rule(31, 1, %0)
      }
    }
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
  }
}
