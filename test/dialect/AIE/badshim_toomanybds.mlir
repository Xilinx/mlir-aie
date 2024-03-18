//===- badmem_toomany_bds.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: 'aie.shim_dma' op has more than 16 blocks

aie.device(xcvc1902) {
  %t1 = aie.tile(2, 0)
  %buf = aie.external_buffer : memref<256xi32>
  %mem = aie.shim_dma(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^bd15)
    ^bd0:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd11
    ^bd11:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd12
    ^bd12:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd13
    ^bd13:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd14
    ^bd14:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd15
    ^bd15:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.next_bd ^bd16
    ^bd16:
      aie.dma_bd(%buf : memref<256xi32>) { len = 256 : i32 }
      aie.end
  }
}
