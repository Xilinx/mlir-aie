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

  %t33 = aie.tile(3, 3)
  %t42 = aie.tile(4, 2)
  %t44 = aie.tile(4, 4)

  %l33_0 = aie.lock(%t33, 0)
  %l33_1 = aie.lock(%t33, 1)
  %l42_0 = aie.lock(%t42, 0)
  %l44_0 = aie.lock(%t44, 0)

  %buf33 = aie.buffer(%t33) : memref<256xi32>
  %buf42 = aie.buffer(%t42) : memref<256xi32>
  %buf44 = aie.buffer(%t44) : memref<256xi32>

  %m33 = aie.mem(%t33) {
      %dmaSt0 = aie.dma_start(MM2S, 0, ^bd0, ^dma0)
    ^dma0:
      %dmaSt1 = aie.dma_start("MM2S", 1, ^bd1, ^end)
    ^bd0:
      aie.use_lock(%l33_0, Acquire, 1)
      aie.dma_bd(%buf33 : memref<256xi32>) { len = 256 : i32 }
      aie.use_lock(%l33_0, Release, 0)
      aie.next_bd ^end
    ^bd1:
      aie.use_lock(%l33_1, Acquire, 0)
      aie.dma_bd(%buf33 : memref<256xi32>) { len = 256 : i32 }
      aie.use_lock(%l33_1, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m42 = aie.mem(%t42) {
      %dmaSt = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l42_0, Acquire, 0)
      aie.dma_bd(%buf42 : memref<256xi32>) { len = 256 : i32 }
      aie.use_lock(%l42_0, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m44 = aie.mem(%t44) {
      %dmaSt = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l44_0, Acquire, 1)
      aie.dma_bd(%buf44 : memref<256xi32>) { len = 256 : i32 }
      aie.use_lock(%l44_0, Release, 0)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %s33 = aie.switchbox(%t33) {
    aie.connect<DMA: 0, East: 0>
    aie.connect<DMA: 1, East: 1>
  }

  %s42 = aie.switchbox(%t42) {
    aie.connect<North: 0, DMA: 0>
  }

  %s44 = aie.switchbox(%t44) {
    aie.connect<South:0, DMA: 0>
  }

  %c33 = aie.core(%t33) {
    aie.use_lock(%l33_1, Acquire, 0)
    aie.use_lock(%l33_0, Acquire, 0)

    // code
    %val0 = arith.constant 16 : i32
    %0 = arith.constant 0 : i32
    aie.put_stream(%0 : i32, %val0 : i32)
    %val1 = aie.get_stream(%0 : i32) : i128
    %val2 = arith.constant 1 : i384
    aie.put_cascade(%val2: i384)

    aie.use_lock(%l33_0, Release, 1)
    aie.use_lock(%l33_1, Release, 1)

    aie.end
  }

  %c42 = aie.core(%t42) {
    aie.use_lock(%l42_0, Acquire, 1)

    // code

    aie.use_lock(%l42_0, Release, 0)
    aie.end
  }

  %c44 = aie.core(%t44) {
    aie.use_lock(%l44_0, Acquire, 1)

    // code

    aie.use_lock(%l44_0, Release, 0)
    aie.end
  }
}
