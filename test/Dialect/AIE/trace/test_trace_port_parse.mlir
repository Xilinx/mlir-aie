//===- test_trace_port_parse.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// CHECK-LABEL: module {
module {
  // CHECK: aie.device(npu1_1col)
  aie.device(npu1_1col) {
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %tile_0_2 = aie.tile(0, 2)
    
    // CHECK: aie.trace @port_trace(%[[TILE]]) {
    aie.trace @port_trace(%tile_0_2) {
      // CHECK: aie.trace.port<0> port = North channel = 1 direction = S2MM
      aie.trace.port<0> port=North channel=1 direction=S2MM
      
      // CHECK: aie.trace.port<1> port = DMA channel = 0 direction = MM2S
      aie.trace.port<1> port=DMA channel=0 direction=MM2S
      
      // CHECK: aie.trace.port<2> port = South channel = 2 direction = S2MM
      aie.trace.port<2> port=South channel=2 direction=S2MM
      
      // CHECK: aie.trace.event <"PORT_RUNNING_0">
      aie.trace.event<"PORT_RUNNING_0">

      // CHECK: aie.trace.event <"PORT_IDLE_1">
      aie.trace.event<"PORT_IDLE_1">
    }
  }
}
