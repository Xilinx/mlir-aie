//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-dma-lowering %s | FileCheck %s

// aiecc and the JIT dispatch bridge both resolve to buildNpuDmaLoweringPipeline.
// If this named pipeline stops resolving, the bridge silently lowers a runtime
// sequence differently from the same design compiled statically.

// The dma_memcpy_nd became a blockwrite of BD words + an address_patch for the
// host buffer, and the dma_wait a sync -- i.e. the whole pass list ran, not
// just the first pass that happens to accept this input.
// CHECK-LABEL: aie.runtime_sequence @seq
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch({{.*}}) {addr = 118788 : ui32, arg_idx = 0 : i32}
// CHECK: aiex.npu.write32
// CHECK: aiex.npu.sync

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, metadata = @alloc0} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @alloc0}
    }
  }
}
