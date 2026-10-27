// RUN: split-file %s %t
// RUN: aie-opt --aie-check-against-image=image=%t/image.mlir %t/image.mlir -o /dev/null
// RUN: not aie-opt --aie-check-against-image=image=%t/image.mlir %t/channel.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHANNEL
// RUN: not aie-opt --aie-check-against-image=image=%t/image.mlir %t/bd-id.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=BD

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CHANNEL: the DMA channel moving external@0: image (0, 0) MM2S 0, this stream (0, 0) MM2S 1
// BD: the BD ids of the chain moving external@0 at (0, 0) MM2S 0: image 0, this stream 1

//--- image.mlir
module {
  aie.device(npu1) {
    %tile = aie.tile(0, 0)
    aie.external_buffer @external : memref<8xi32>
    aie.shim_dma(%tile) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      %buffer = aie.get_buffer @external : memref<8xi32>
      aie.dma_bd(%buffer : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

//--- channel.mlir
module {
  aie.device(npu1) {
    %tile = aie.tile(0, 0)
    aie.external_buffer @external : memref<8xi32>
    aie.shim_dma(%tile) {
      aie.dma_start(MM2S, 1, ^bd, ^end)
    ^bd:
      %buffer = aie.get_buffer @external : memref<8xi32>
      aie.dma_bd(%buffer : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

//--- bd-id.mlir
module {
  aie.device(npu1) {
    %tile = aie.tile(0, 0)
    aie.external_buffer @external : memref<8xi32>
    aie.shim_dma(%tile) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      %buffer = aie.get_buffer @external : memref<8xi32>
      aie.dma_bd(%buffer : memref<8xi32> offset = 0 len = 8) {bd_id = 1 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
