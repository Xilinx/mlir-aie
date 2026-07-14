//===- test_lock6.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock6 {
// CHECK:         aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(5, 5)
// CHECK:           %[[VAL_1:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32}
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:           %[[VAL_3:.*]] = aie.tile(4, 4)
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_3]], 0) {init = 0 : i32}
// CHECK:           %[[VAL_5:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_5]], 0) {init = 0 : i32}
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) : memref<256xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_3]]) : memref<256xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:           aiex.token(0) {sym_name = "token0"}
// CHECK:           aiex.token(0) {sym_name = "token1"}
// CHECK:           %[[VAL_11:.*]] = aie.mem(%[[VAL_5]]) {
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aiex.useToken @token0(Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<256xi32> len = {{.*}})
// CHECK:             aiex.useToken @token0(Release, 2)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_14:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aiex.useToken @token1(Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<256xi32> len = {{.*}})
// CHECK:             aiex.useToken @token1(Release, 2)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(S2MM, 0, ^bb2, ^bb1)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
// CHECK:           ^bb2:
// CHECK:             aiex.useToken @token0(Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<256xi32> len = {{.*}})
// CHECK:             aiex.useToken @token0(Release, 2)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb3:
// CHECK:             aiex.useToken @token1(Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<256xi32> len = {{.*}})
// CHECK:             aiex.useToken @token1(Release, 2)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.core(%[[VAL_5]]) {
// CHECK:             aiex.useToken @token0(Acquire, 0)
// CHECK:             aiex.useToken @token0(Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:             aiex.useToken @token1(Acquire, 0)
// CHECK:             aiex.useToken @token1(Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             aiex.useToken @token1(Acquire, 2)
// CHECK:             aiex.useToken @token0(Acquire, 2)
// CHECK:             aiex.useToken @token0(Release, 3)
// CHECK:             aiex.useToken @token1(Release, 3)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.flow(%[[VAL_5]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_0]], DMA : 1)
// CHECK:         }
// CHECK:       }

module @test_lock6 {
 aie.device(xcvc1902) {
  %t55 = aie.tile(5, 5)
  %t44 = aie.tile(4, 4)
  %t33 = aie.tile(3, 3)

  %buf33   = aie.buffer(%t33) : memref<256xi32>
  %buf44   = aie.buffer(%t44) : memref<256xi32>
  %buf55_0 = aie.buffer(%t55) : memref<256xi32>
  %buf55_1 = aie.buffer(%t55) : memref<256xi32>

  aiex.token(0) {sym_name = "token0"}
  aiex.token(0) {sym_name = "token1"}

  %m33 = aie.mem(%t33) {
      %dmaSt = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aiex.useToken @token0(Acquire, 1)
      aie.dma_bd(%buf33 : memref<256xi32> len = 256)
      aiex.useToken @token0(Release, 2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m44 = aie.mem(%t44) {
      %dmaSt = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aiex.useToken @token1(Acquire, 1)
      aie.dma_bd(%buf44 : memref<256xi32> len = 256)
      aiex.useToken @token1(Release, 2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m55 = aie.mem(%t55) {
      %dmaSt0 = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dmaSt1 = aie.dma_start("S2MM", 1, ^bd1, ^end)
    ^bd0:
      aiex.useToken @token0(Acquire, 1)
      aie.dma_bd(%buf55_0 : memref<256xi32> len = 256)
      aiex.useToken @token0(Release, 2)
      aie.next_bd ^end
    ^bd1:
      aiex.useToken @token1(Acquire, 1)
      aie.dma_bd(%buf55_1 : memref<256xi32> len = 256)
      aiex.useToken @token1(Release, 2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %c33 = aie.core(%t33) {
    aiex.useToken @token0(Acquire, 0)
    aiex.useToken @token0(Release, 1)
    aie.end
  }

  %c44 = aie.core(%t44) {
    aiex.useToken @token1(Acquire, 0)
    aiex.useToken @token1(Release, 1)
    aie.end
  }

  %c55 = aie.core(%t55) {
    aiex.useToken @token1(Acquire, 2)
    aiex.useToken @token0(Acquire, 2)
    aiex.useToken @token0(Release, 3)
    aiex.useToken @token1(Release, 3)
    aie.end
  }

  aie.flow(%t33, DMA : 0, %t55, DMA : 0)
  aie.flow(%t44, DMA : 0, %t55, DMA : 1)
 }
}
