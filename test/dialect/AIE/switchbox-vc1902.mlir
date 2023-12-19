//===- switchbox-vc1902.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s

module {
  AIE.device(xcvc1902) {
    %20 = AIE.tile(2, 0) // Shim-NOC
    AIE.switchbox(%20) {
      AIE.connect<East: 0, East: 0> // Feedback OK
      AIE.connect<South: 0, South: 2> // Bounce OK
      AIE.connect<FIFO: 1, West: 2> // Two fifo connections
      AIE.connect<South: 5, North: 5> // 6 northgoing connections
      AIE.connect<East: 3, West: 3> // 4 westgoing connections
      AIE.connect<North: 3, South: 3> // 4 southgoing connections
      AIE.connect<West: 3, East: 3> // 4 eastgoing connections
      AIE.connect<Trace: 0, South: 1>
    }

    %30 = AIE.tile(3, 0) // Shim-NOC
    AIE.shim_mux(%30) {
      AIE.connect<DMA: 0, North: 3>
      AIE.connect<DMA: 1, North: 7>
      AIE.connect<North: 0, PLIO: 0>
      AIE.connect<North: 1, PLIO: 1>
      AIE.connect<North: 2, DMA: 0>
      AIE.connect<North: 3, DMA: 1>
      AIE.connect<North: 4, PLIO: 4>
      AIE.connect<North: 5, PLIO: 5>
    }

    %40 = AIE.tile(4, 0) // Shim-PL tile
    AIE.switchbox(%40) {
      AIE.connect<East: 0, East: 0> // Feedback OK
      AIE.connect<South: 0, South: 2> // Bounce OK
      AIE.connect<FIFO: 1, West: 2> // Two fifo connections
      AIE.connect<South: 5, North: 5> // 6 northgoing connections
      AIE.connect<East: 3, West: 3> // 4 westgoing connections
      AIE.connect<North: 3, South: 3> // 4 southgoing connections
      AIE.connect<West: 3, East: 3> // 4 eastgoing connections
      AIE.connect<Trace: 0, South: 1>
    }

    %60 = AIE.tile(6, 0) // Shim-NOC
    AIE.shim_mux(%60) {
      AIE.connect<PLIO: 0, North: 0>
      AIE.connect<PLIO: 1, North: 1>
      AIE.connect<PLIO: 2, North: 2>
      AIE.connect<PLIO: 3, North: 3>
      AIE.connect<PLIO: 4, North: 4>
      AIE.connect<PLIO: 5, North: 5>
      AIE.connect<PLIO: 6, North: 6>
      AIE.connect<PLIO: 7, North: 7>
      AIE.connect<North: 0, PLIO: 0>
      AIE.connect<North: 1, PLIO: 1>
      AIE.connect<North: 2, PLIO: 2>
      AIE.connect<North: 3, PLIO: 3>
      AIE.connect<North: 4, PLIO: 4>
      AIE.connect<North: 5, PLIO: 5>
    }
    AIE.shim_mux(%60) {
      AIE.connect<NOC: 0, North: 7>
      AIE.connect<NOC: 1, North: 6>
      AIE.connect<NOC: 2, North: 3>
      AIE.connect<NOC: 3, North: 2>
      AIE.connect<North: 0, PLIO: 0>
      AIE.connect<North: 1, PLIO: 1>
      AIE.connect<North: 2, NOC: 2>
      AIE.connect<North: 3, NOC: 3>
      AIE.connect<North: 4, NOC: 4>
      AIE.connect<North: 5, NOC: 5>
    }

    %01 = AIE.tile(1, 1)
    AIE.switchbox(%01) {
      AIE.connect<East: 0, East: 0> // Feedback OK
      AIE.connect<South: 0, South: 2> // Bounce OK
      AIE.connect<DMA: 1, East: 1>  // Two core connections
      AIE.connect<Core: 1, East: 2> // Two core connections
      AIE.connect<FIFO: 1, North: 2> // Two fifo connections
      AIE.connect<South: 5, North: 5> // 6 northgoing connections
      AIE.connect<East: 3, West: 3> // 4 westgoing connections
      AIE.connect<North: 3, South: 3> // 4 southgoing connections
      AIE.connect<West: 3, East: 3> // 4 eastgoing connections
    }
  }
}
