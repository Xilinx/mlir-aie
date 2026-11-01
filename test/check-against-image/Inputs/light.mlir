// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The design an image is built from: two shim logical tiles feeding one core,
// with a runtime sequence light enough that they share one shim.
module {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(0, 2)
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in_a(%shim_a, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo @in_b(%shim_b, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @in_a(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_a(Consume, 1)
      %1 = aie.objectfifo.acquire @in_b(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%a: memref<16777216xbf16>, %b: memref<16777216xbf16>) {
      %ta = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 0 len = 4096)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ta)
      %tb = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<16777216xbf16> offset = 0 len = 4096)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tb)
      aiex.dma_await_task(%ta)
      aiex.dma_await_task(%tb)
    }
  }
}
