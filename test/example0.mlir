//===- example0.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// Goal of AIE dialect modeling: ensure the legality of Lock + DMA + Memory accesses
// and StreamSwitch connectivity
// This is a physical netlist of mapping code to multiple AIE cores in the AIE array
//
// The physical netlist will get translated to XAIE* calls (from the ARM host) to configure
// the AIE array (Core, StreamSwitch, Lock, DMMA setups)
// and the code in the CoreModule region will get translated to AIE core instrinsics

// CHECK-LABEL: module @example0 {
// CHECK:       }

module @example0 {

  // Odd  AIE rows: DMem on the East
  // Even AIE rows: DMem on the West

  // (2, 4) (3, 4) (4, 4) (5, 4)
  // (2, 3) (3, 3) (4, 3) (5, 3)
  // (2, 2) (3, 2) (4, 2) (5, 2)

  %t33 = AIE.tile(3, 3)
  %t42 = AIE.tile(4, 2)
  %t44 = AIE.tile(4, 4)

  %l33_0 = AIE.lock(%t33, 0)
  %l33_1 = AIE.lock(%t33, 1)
  %l42_0 = AIE.lock(%t42, 0)
  %l44_0 = AIE.lock(%t44, 0)

  %buf33 = AIE.buffer(%t33) : memref<256xi32>
  %buf42 = AIE.buffer(%t42) : memref<256xi32>
  %buf44 = AIE.buffer(%t44) : memref<256xi32>

  %m33 = AIE.mem(%t33) {
      %dmaSt0 = AIE.dmaStart(MM2S0, ^bd0, ^dma0)
    ^dma0:
      %dmaSt1 = AIE.dmaStart("MM2S1", ^bd1, ^end)
    ^bd0:
      AIE.useLock(%l33_0, Acquire, 1, 0)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, Release, 0, 0)
      br ^end
    ^bd1:
      AIE.useLock(%l33_1, Acquire, 0, 0)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_1, Release, 1, 0)
      br ^end
    ^end:
      AIE.end
  }

  %m42 = AIE.mem(%t42) {
      %dmaSt = AIE.dmaStart(S2MM0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l42_0, Acquire, 0, 0)
      AIE.dmaBd(<%buf42 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l42_0, Release, 1, 0)
      br ^end
    ^end:
      AIE.end
  }

  %m44 = AIE.mem(%t44) {
      %dmaSt = AIE.dmaStart(S2MM0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l44_0, Acquire, 1, 0)
      AIE.dmaBd(<%buf44 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l44_0, Release, 0, 0)
      br ^end
    ^end:
      AIE.end
  }

  %s33 = AIE.switchbox(%t33) {
    AIE.connect<DMA: 0, East: 0>
    AIE.connect<DMA: 1, East: 1>
  }

  %s42 = AIE.switchbox(%t42) {
    AIE.connect<North: 0, DMA: 0>
  }

  %s44 = AIE.switchbox(%t44) {
    AIE.connect<South:0, DMA: 0>
  }

  %c33 = AIE.core(%t33) {
    AIE.useLock(%l33_1, Acquire, 0, 0)
    AIE.useLock(%l33_0, Acquire, 0, 0)

    // code
    %val0 = constant 16 : i32
    %0 = constant 0 : i32
    AIE.putStream(%0 : i32, %val0 : i32)
    %val1 = AIE.getStream(%0 : i32) : i128
    %val2 = constant 1 : i384
    AIE.putCascade(%val2: i384)

    AIE.useLock(%l33_0, Release, 1, 0)
    AIE.useLock(%l33_1, Release, 1, 0)

    AIE.end
  }

  %c42 = AIE.core(%t42) {
    AIE.useLock(%l42_0, Acquire, 1, 0)

    // code

    AIE.useLock(%l42_0, Release, 0, 0)
    AIE.end
  }

  %c44 = AIE.core(%t44) {
    AIE.useLock(%l44_0, Acquire, 1, 0)

    // code

    AIE.useLock(%l44_0, Release, 0, 0)
    AIE.end
  }
}
