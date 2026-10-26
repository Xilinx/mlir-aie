//===- bad_flow_port.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2_1col) {
  %t02 = aie.tile(0, 2)
  %t03 = aie.tile(0, 3)
  aie.packet_flow(3) {
    aie.packet_source<%t02, DMA : 0>
    // expected-error@+1 {{destination Core:1 does not exist: tile (0, 3) has 1 Core port out of its stream switch}}
    aie.packet_dest<%t03, Core : 1>
  }
}

// -----

aie.device(npu2_1col) {
  %t02 = aie.tile(0, 2)
  %t03 = aie.tile(0, 3)
  aie.packet_flow(3) {
    // expected-error@+1 {{source DMA:2 does not exist: tile (0, 2) has 2 DMA ports into its stream switch}}
    aie.packet_source<%t02, DMA : 2>
    aie.packet_dest<%t03, DMA : 0>
  }
}

// -----

aie.device(npu1_1col) {
  %t01 = aie.tile(0, 1)
  %t02 = aie.tile(0, 2)
  // expected-error@+1 {{destination Core:0 does not exist: tile (0, 1) has 0 Core ports out of its stream switch}}
  aie.flow(%t02, DMA : 0, %t01, Core : 0)
}

// -----

aie.device(npu1_1col) {
  %t00 = aie.tile(0, 0)
  %t02 = aie.tile(0, 2)
  // expected-error@+1 {{source DMA:2 does not exist: tile (0, 0) has 2 DMA ports into its stream switch}}
  aie.flow(%t00, DMA : 2, %t02, DMA : 0)
}
