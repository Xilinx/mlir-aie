//===- lower_dma.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="tilecol=3 tilerow=3" %s | FileCheck %s

// CHECK:    call @llvm.aie.lock.acquire.reg({{.*}}, %c0_i32) : (i32, i32) -> ()
// CHECK:    call @llvm.aie.put.ms(%c0_i32_0, %c16_i32) : (i32, i32) -> ()
// CHECK:    {{.*}} = call @llvm.aie.get.wss(%c0_i32_0) : (i32) -> i128
// CHECK:    call @llvm.aie.put.mcd(%c1_i384) : (i384) -> ()
// CHECK:    call @llvm.aie.lock.release.reg({{.*}}, %c1_i32) : (i32, i32) -> ()

module @example0 {
 aie.device(xcvc1902) {

  // Odd  AIE rows: DMem on the East
  // Even AIE rows: DMem on the West

  // (2, 4) (3, 4) (4, 4) (5, 4)
  // (2, 3) (3, 3) (4, 3) (5, 3)
  // (2, 2) (3, 2) (4, 2) (5, 2)

  %t11 = aie.tile(1, 1)
  %t33 = aie.tile(3, 3)
  %t43 = aie.tile(4, 3)

  %l33_0 = aie.lock(%t33, 0)
  %l33_1 = aie.lock(%t33, 1)
  %l43_0 = aie.lock(%t43, 0)

  %buf33 = aie.buffer(%t33) { sym_name = "a" } : memref<256xi32>
  %buf43 = aie.buffer(%t43) { sym_name = "b" } : memref<256xi32>

  %m33 = aie.mem(%t33) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
      %dmaSt0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%l33_0, Acquire, %c1_ul1)
      aie.dma_bd(%buf33 : memref<256xi32> offset = 0 len = 256)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(%l33_0, Release, %c0_ul2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m43 = aie.mem(%t43) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
      %dmaSt = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(%l43_0, Acquire, %c0_ul3)
      aie.dma_bd(%buf43 : memref<256xi32> offset = 0 len = 256)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%l43_0, Release, %c1_ul4)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %s33 = aie.switchbox(%t33) {
    aie.connect<DMA: 0, North: 0>
  }

  %s43 = aie.switchbox(%t43) {
    aie.connect<South: 0, DMA: 0>
  }

  %c33 = aie.core(%t33) {
    %c0_ul5 = arith.constant 0 : i32
    aie.use_lock(%l33_0, Acquire, %c0_ul5)
    // code
    %val0 = arith.constant 16 : i32
    %0 = arith.constant 0 : i32
    aie.put_stream(%0 : i32, %val0 : i32)
    %val1 = aie.get_stream(%0 : i32) : i128
    %val2 = arith.constant 1 : i384
    aie.put_cascade(%val2: i384)
    %c1_ul6 = arith.constant 1 : i32
    aie.use_lock(%l33_0, Release, %c1_ul6)
    aie.end
  }

  %c43 = aie.core(%t43) {
    %c1_ul7 = arith.constant 1 : i32
    aie.use_lock(%l43_0, Acquire, %c1_ul7)

    // code

    %c0_ul8 = arith.constant 0 : i32
    aie.use_lock(%l43_0, Release, %c0_ul8)
    aie.end
  }
 }
}
