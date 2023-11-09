//===- packet_header.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s | FileCheck %s

//CHECK: x = 0;
//CHECK: y = 0;
//CHECK: /* drop_header */ XAIE_SS_PKT_DONOT_DROP_HEADER
//CHECK: x = 0;
//CHECK: y = 1;
//CHECK: /* drop_header */ XAIE_SS_PKT_DROP_HEADER
//CHECK: /* drop_header */ XAIE_SS_PKT_DONOT_DROP_HEADER
//CHECK: x = 0;
//CHECK: y = 2;
//CHECK: /* drop_header */ XAIE_SS_PKT_DONOT_DROP_HEADER

module {
  AIE.device(ipu) {
    %0 = AIE.tile(0, 0)
    %1 = AIE.shimmux(%0) {
      AIE.connect<DMA : 0, North : 3>
    }
    %2 = AIE.switchbox(%0) {
      %7 = AIE.amsel<0> (0)
      %8 = AIE.masterset(North : 0, %7)
      AIE.packetrules(South : 3) {
        AIE.rule(31, 1, %7)
      }
    }
    %3 = AIE.tile(0, 1)
    %4 = AIE.switchbox(%3) {
      %7 = AIE.amsel<0> (0)
      %8 = AIE.amsel<0> (1)
      %9 = AIE.masterset(DMA : 0, %7)
      %10 = AIE.masterset(North : 0, %8)
      AIE.packetrules(DMA : 0) {
        AIE.rule(31, 2, %8)
      }
      AIE.packetrules(South : 0) {
        AIE.rule(31, 1, %7)
      }
    }
    %5 = AIE.tile(0, 2)
    %6 = AIE.switchbox(%5) {
      %7 = AIE.amsel<0> (0)
      %8 = AIE.masterset(DMA : 0, %7) {keep_pkt_header = true}
      AIE.packetrules(South : 0) {
        AIE.rule(31, 2, %7)
      }
    }
    AIE.wire(%1 : North, %2 : South)
    AIE.wire(%0 : DMA, %1 : DMA)
    AIE.wire(%3 : Core, %4 : Core)
    AIE.wire(%3 : DMA, %4 : DMA)
    AIE.wire(%2 : North, %4 : South)
    AIE.wire(%5 : Core, %6 : Core)
    AIE.wire(%5 : DMA, %6 : DMA)
    AIE.wire(%4 : North, %6 : South)
    AIE.packet_flow(1) {
      AIE.packet_source<%0, DMA : 0>
      AIE.packet_dest<%3, DMA : 0>
    }
    AIE.packet_flow(2) {
      AIE.packet_source<%3, DMA : 0>
      AIE.packet_dest<%5, DMA : 0>
    }
  }
}
