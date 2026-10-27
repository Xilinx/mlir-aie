//===- test_lock3.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock3 {
// CHECK:         aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(4, 4)
// CHECK:           aie.lock @[[VAL_1:.*]](%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:           aie.lock @[[VAL_3:.*]](%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:           aie.buffer @[[VAL_4:.*]](%[[VAL_2]]) : memref<256xi32>
// CHECK:           aie.buffer @[[VAL_5:.*]](%[[VAL_0]]) : memref<256xi32>
// CHECK:           aiex.token(0) {sym_name = "token0"}
// CHECK:           %[[VAL_6:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_7:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aiex.useToken @token0(Acquire, 1)
// CHECK:             %[[VAL_4_ACCESS_0:.*]] = aie.get_buffer @[[VAL_4]] : memref<256xi32>
// CHECK:             aie.dma_bd(%[[VAL_4_ACCESS_0]] : memref<256xi32> len = {{.*}})
// CHECK:             aiex.useToken @token0(Release, 2)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_9:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aiex.useToken @token0(Acquire, 1)
// CHECK:             %[[VAL_5_ACCESS_1:.*]] = aie.get_buffer @[[VAL_5]] : memref<256xi32>
// CHECK:             aie.dma_bd(%[[VAL_5_ACCESS_1]] : memref<256xi32> len = {{.*}})
// CHECK:             aiex.useToken @token0(Release, 2)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             aiex.useToken @token0(Acquire, 0)
// CHECK:             aiex.useToken @token0(Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             aiex.useToken @token0(Acquire, 2)
// CHECK:             aiex.useToken @token0(Release, 3)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:         }
// CHECK:       }

module @test_lock3 {
 aie.device(xcvc1902) {
  %t44 = aie.tile(4, 4)
  %t33 = aie.tile(3, 3)
  aie.buffer @buf33(%t33) : memref<256xi32>
  aie.buffer @buf44(%t44) : memref<256xi32>
  aiex.token(0) {sym_name = "token0"}
  %m33 = aie.mem(%t33) {
      %dmaSt = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aiex.useToken @token0(Acquire, 1)
      %buf33 = aie.get_buffer @buf33 : memref<256xi32>
      aie.dma_bd(%buf33 : memref<256xi32> len = 256)
      aiex.useToken @token0(Release, 2)
      aie.next_bd ^end
    ^end:
      aie.end
  }
  %m44 = aie.mem(%t44) {
      %dmaSt = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aiex.useToken @token0(Acquire, 1)
      %buf44 = aie.get_buffer @buf44 : memref<256xi32>
      aie.dma_bd(%buf44 : memref<256xi32> len = 256)
      aiex.useToken @token0(Release, 2)
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
    aiex.useToken @token0(Acquire, 2)
    aiex.useToken @token0(Release, 3)
    aie.end
  }
  aie.flow(%t33, DMA : 0, %t44, DMA : 0)
 }
}
