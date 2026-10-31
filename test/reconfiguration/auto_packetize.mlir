// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-split --aie-create-control-overlay="adaptive=false" %s | FileCheck %s --check-prefix=PLAN
// RUN: aie-opt --aie-objectfifo-split --aie-create-control-overlay="adaptive=false" --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s --check-prefix=ALLOC
// RUN: not aie-opt --aie-objectfifo-split --aie-create-control-overlay="adaptive=false share-channels=false" --aie-assign-packet-ids --aie-objectfifo-allocate %s 2>&1 | FileCheck %s --check-prefix=OFF

// PLAN-LABEL: @shared
// PLAN: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// PLAN-SAME: aie.fixed
// PLAN-SAME: aie.shareable_with = [@a_prod_dma, @b_prod_dma]
// PLAN-LABEL: @ctrl_pkt_overlay
// ALLOC-LABEL: @shared
// ALLOC-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%{{.*}}, MM2S, 0)
// ALLOC-DAG: aie.shim_dma_allocation @{{[ab]}}_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// ALLOC-DAG: aie.shim_dma_allocation @{{[ab]}}_shim_alloc(%{{.*}}, MM2S, 1)
// ALLOC-NOT: _relay
// OFF: number of output DMA channel exceeded

module {
  aie.device(npu2_1col) @shared {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @a(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}
