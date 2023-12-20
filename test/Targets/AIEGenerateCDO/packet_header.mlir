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
  aie.device(ipu) {
    %0 = aie.tile(0, 0)
    %1 = aie.shim_mux(%0) {
      aie.connect<DMA : 0, North : 3>
    }
    %2 = aie.switchbox(%0) {
      %7 = aie.amsel<0> (0)
      %8 = aie.masterset(North : 0, %7)
      aie.packet_rules(South : 3) {
        aie.rule(31, 1, %7)
      }
    }
    %3 = aie.tile(0, 1)
    %4 = aie.switchbox(%3) {
      %7 = aie.amsel<0> (0)
      %8 = aie.amsel<0> (1)
      %9 = aie.masterset(DMA : 0, %7)
      %10 = aie.masterset(North : 0, %8)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 2, %8)
      }
      aie.packet_rules(South : 0) {
        aie.rule(31, 1, %7)
      }
    }
    %5 = aie.tile(0, 2)
    %6 = aie.switchbox(%5) {
      %7 = aie.amsel<0> (0)
      %8 = aie.masterset(DMA : 0, %7) {keep_pkt_header = true}
      aie.packet_rules(South : 0) {
        aie.rule(31, 2, %7)
      }
    }
    aie.wire(%1 : North, %2 : South)
    aie.wire(%0 : DMA, %1 : DMA)
    aie.wire(%3 : Core, %4 : Core)
    aie.wire(%3 : DMA, %4 : DMA)
    aie.wire(%2 : North, %4 : South)
    aie.wire(%5 : Core, %6 : Core)
    aie.wire(%5 : DMA, %6 : DMA)
    aie.wire(%4 : North, %6 : South)
    aie.packet_flow(1) {
      aie.packet_source<%0, DMA : 0>
      aie.packet_dest<%3, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%3, DMA : 0>
      aie.packet_dest<%5, DMA : 0>
    }
  }
}
