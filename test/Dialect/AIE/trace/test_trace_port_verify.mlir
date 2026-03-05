//===- test_trace_port_verify.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    
    aie.trace @bad_slot(%tile_0_2) {
      // expected-error @+1 {{attribute 'slot' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 7}}
      aie.trace.port<8> port=North channel=1 direction=S2MM
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    aie.trace @dma_channel_oob(%tile_0_2) {
      // expected-error @+1 {{invalid stream switch port configuration for tile (0, 2)}}
      aie.trace.port<0> port=DMA channel=7 direction=MM2S
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    
    aie.trace @duplicate_slot(%tile_0_2) {
      // expected-error @+1 {{duplicate port slot 0 in trace duplicate_slot}}
      aie.trace.port<0> port=North channel=1 direction=S2MM
      aie.trace.port<0> port=DMA channel=0 direction=MM2S
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    
    aie.trace @negative_channel(%tile_0_2) {
      // expected-error @+1 {{invalid stream switch port configuration for tile (0, 2)}}
      aie.trace.port<0> port=North channel=-1 direction=S2MM
    }
  }
}
