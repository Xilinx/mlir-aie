//===- switchbox-vc2802.mlir -----------------------------------*- MLIR -*-===//
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
  aie.device(xcve2802) {
    %20 = aie.tile(2, 0) // Shim-NOC
    aie.switchbox(%20) {
      aie.connect<East: 0, East: 0> // Feedback OK
      aie.connect<South: 0, South: 2> // Bounce OK
      aie.connect<FIFO: 0, West: 2> // One fifo connections
      aie.connect<South: 5, North: 5> // 6 northgoing connections
      aie.connect<East: 3, West: 3> // 4 westgoing connections
      aie.connect<North: 3, South: 3> // 4 southgoing connections
      aie.connect<West: 3, East: 3> // 4 eastgoing connections
      aie.connect<Trace: 0, South: 1>
      aie.connect<Ctrl: 0, North: 2>
      aie.connect<North: 2, Ctrl: 0>
    }

    %30 = aie.tile(3, 0) // Shim-NOC
    aie.shim_mux(%30) {
      aie.connect<DMA: 0, North: 3>
      aie.connect<DMA: 1, North: 7>
      aie.connect<North: 0, PLIO: 0>
      aie.connect<North: 1, PLIO: 1>
      aie.connect<North: 2, DMA: 0>
      aie.connect<North: 3, DMA: 1>
      aie.connect<North: 4, PLIO: 4>
      aie.connect<North: 5, PLIO: 5>
    }

    %40 = aie.tile(4, 0) // Shim-PL tile
    aie.switchbox(%40) {
      aie.connect<East: 0, East: 0> // Feedback OK
      aie.connect<South: 0, South: 2> // Bounce OK
      aie.connect<FIFO: 0, West: 2> // One fifo connections
      aie.connect<South: 5, North: 5> // 6 northgoing connections
      aie.connect<East: 3, West: 3> // 4 westgoing connections
      aie.connect<North: 3, South: 3> // 4 southgoing connections
      aie.connect<West: 3, East: 3> // 4 eastgoing connections
      aie.connect<Trace: 0, South: 1>
    }

    %60 = aie.tile(6, 0) // Shim-NOC
    aie.shim_mux(%60) {
      aie.connect<PLIO: 0, North: 0>
      aie.connect<PLIO: 1, North: 1>
      aie.connect<PLIO: 2, North: 2>
      aie.connect<PLIO: 3, North: 3>
      aie.connect<PLIO: 4, North: 4>
      aie.connect<PLIO: 5, North: 5>
      aie.connect<PLIO: 6, North: 6>
      aie.connect<PLIO: 7, North: 7>
      aie.connect<North: 0, PLIO: 0>
      aie.connect<North: 1, PLIO: 1>
      aie.connect<North: 2, PLIO: 2>
      aie.connect<North: 3, PLIO: 3>
      aie.connect<North: 4, PLIO: 4>
      aie.connect<North: 5, PLIO: 5>
    }
    aie.shim_mux(%60) {
      aie.connect<NOC: 0, North: 7>
      aie.connect<NOC: 1, North: 6>
      aie.connect<NOC: 2, North: 3>
      aie.connect<NOC: 3, North: 2>
      aie.connect<North: 0, PLIO: 0>
      aie.connect<North: 1, PLIO: 1>
      aie.connect<North: 2, NOC: 2>
      aie.connect<North: 3, NOC: 3>
      aie.connect<North: 4, NOC: 4>
      aie.connect<North: 5, NOC: 5>
    }

    %01 = aie.tile(0, 1) // mem tile
    aie.switchbox(%01) {
      aie.connect<South: 0, South: 0> // Feedback OK
      aie.connect<South: 0, South: 2> // Bounce OK
      aie.connect<DMA: 5, North: 1>  // 5 DMA connections
      aie.connect<South: 5, North: 5> // 6 northgoing connections
      aie.connect<North: 3, South: 3> // 4 southgoing connections
      aie.connect<Trace: 0, South: 1>
      aie.connect<Ctrl: 0, North: 2>
      aie.connect<North: 2, Ctrl: 0>
    }

    %03 = aie.tile(1, 3) // core tile
    aie.switchbox(%03) {
      aie.connect<East: 0, East: 0> // Feedback OK
      aie.connect<South: 0, South: 2> // Bounce OK
      aie.connect<DMA: 1, East: 1>  // Two DMA connections
      aie.connect<Core: 0, East: 2> // One core connections
      aie.connect<FIFO: 0, West: 2> // One fifo connections
      aie.connect<South: 5, North: 5> // 6 northgoing connections
      aie.connect<East: 3, West: 3> // 4 westgoing connections
      aie.connect<North: 3, South: 3> // 4 southgoing connections
      aie.connect<West: 3, East: 3> // 4 eastgoing connections
      aie.connect<Ctrl: 0, North: 2>
      aie.connect<North: 2, Ctrl: 0>
    }
  }
}
