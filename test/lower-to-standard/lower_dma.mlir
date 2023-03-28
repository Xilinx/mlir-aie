//===- lower_dma.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="tilecol=3 tilerow=3" %s | FileCheck %s

// CHECK:    call @llvm.aie.lock.acquire.reg({{.*}}, %c0_i32) : (i32, i32) -> ()
// CHECK:    call @llvm.aie.put.ms(%c16_i32, %c0_i32_0) : (i32, i32) -> ()
// CHECK:    {{.*}} = call @llvm.aie.get.wss(%c0_i32_0) : (i32) -> i128
// CHECK:    call @llvm.aie.put.mcd(%c1_i384) : (i384) -> ()
// CHECK:    call @llvm.aie.lock.release.reg({{.*}}, %c1_i32) : (i32, i32) -> ()

module @example0 {
 AIE.device(xcvc1902) {

  // Odd  AIE rows: DMem on the East
  // Even AIE rows: DMem on the West

  // (2, 4) (3, 4) (4, 4) (5, 4)
  // (2, 3) (3, 3) (4, 3) (5, 3)
  // (2, 2) (3, 2) (4, 2) (5, 2)

  %t11 = AIE.tile(1, 1)
  %t33 = AIE.tile(3, 3)
  %t43 = AIE.tile(4, 3)

  %l33_0 = AIE.lock(%t33, 0)
  %l33_1 = AIE.lock(%t33, 1)
  %l43_0 = AIE.lock(%t43, 0)

  %buf33 = AIE.buffer(%t33) { sym_name = "a" } : memref<256xi32>
  %buf43 = AIE.buffer(%t43) { sym_name = "b" } : memref<256xi32>

  %m33 = AIE.mem(%t33) {
      %dmaSt0 = AIE.dmaStart(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l33_0, Acquire, 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, Release, 0)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %m43 = AIE.mem(%t43) {
      %dmaSt = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l43_0, Acquire, 0)
      AIE.dmaBd(<%buf43 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l43_0, Release, 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %s33 = AIE.switchbox(%t33) {
    AIE.connect<DMA: 0, North: 0>
  }

  %s43 = AIE.switchbox(%t43) {
    AIE.connect<South: 0, DMA: 0>
  }

  %c33 = AIE.core(%t33) {
    AIE.useLock(%l33_0, Acquire, 0)
    // code
    %val0 = arith.constant 16 : i32
    %0 = arith.constant 0 : i32
    AIE.putStream(%0 : i32, %val0 : i32)
    %val1 = AIE.getStream(%0 : i32) : i128
    %val2 = arith.constant 1 : i384
    AIE.putCascade(%val2: i384)
    AIE.useLock(%l33_0, Release, 1)
    AIE.end
  }

  %c43 = AIE.core(%t43) {
    AIE.useLock(%l43_0, Acquire, 1)

    // code

    AIE.useLock(%l43_0, Release, 0)
    AIE.end
  }
 }
}
