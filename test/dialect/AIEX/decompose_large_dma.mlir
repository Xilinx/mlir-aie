//===- decompose_large_dma.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for aie-decompose-large-dma-bd: oversized non-contiguous ND DMA
// patterns are split into hardware-legal aiex.npu.dma_memcpy_nd ops.
//
//===----------------------------------------------------------------------===//


// -----

// Test 1: a non-contiguous 1920x1080 i32 access (row stride 1921 != 1920, so it
// cannot be linearized) has d0=1920 and d1=1080, both exceeding the 10-bit
// shim wrap limit (1023). The pass rewrites it into legal sub-transfers; the
// original oversized "1080, 1920]" size list must no longer appear.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=DECOMPOSE

// DECOMPOSE-LABEL: @decompose_2d
// DECOMPOSE:         aiex.npu.dma_memcpy_nd
// DECOMPOSE-NOT:     1080, 1920]
module {
  aie.device(npu1) {
    aie.runtime_sequence @decompose_2d(%in : memref<1920x1080xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0, 0, 0, 0][1, 1, 1080, 1920][0, 0, 1921, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<1920x1080xi32>
    }
    %tile00 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile00, MM2S, 0)
  }
}


// -----

// Test 2: end-to-end through aie-dma-to-npu after decomposition. The lowering
// must succeed (no "exceeds the [0:1023] range" diagnostic) and emit blockwrite
// configuration for the resulting legal BDs.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd,aie-dma-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWER

// LOWER-LABEL: @lower_2d
// LOWER-NOT:     exceeds the [0:1023] range
// LOWER:         aiex.npu.blockwrite
module {
  aie.device(npu1) {
    aie.runtime_sequence @lower_2d(%in : memref<1920x1080xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0, 0, 0, 0][1, 1, 1080, 1920][0, 0, 1921, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<1920x1080xi32>
    }
    %tile00 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile00, MM2S, 0)
  }
}


// -----

// Test 3: a small already-legal op is left unchanged (no spurious rewrite).
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=UNCHANGED

// UNCHANGED-LABEL: @small_unchanged
// UNCHANGED:         aiex.npu.dma_memcpy_nd
// UNCHANGED-SAME:      [0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @small_unchanged(%a : memref<8xi32>) {
      aiex.npu.dma_memcpy_nd (%a[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1])
        { metadata = @fifo, id = 0 : i64 } : memref<8xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @fifo (%tile_0_0, MM2S, 0)
  }
}


// -----

// Test 4: SLICE — a prime outer dimension (1031 > 1023) cannot be factored, so
// it is split into multiple in-order dma_memcpy_nd ops (each a contiguous slice
// of the oversized dimension, with adjusted offset).
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=SLICE

// SLICE-LABEL: @slice_memcpy
// SLICE:         aiex.npu.dma_memcpy_nd
// SLICE-SAME:      [0, 0, 0, 0][1, 1, 1023, 2][0, 0, 3, 1]
// SLICE:         aiex.npu.dma_memcpy_nd
// SLICE-SAME:      [0, 0, 1023, 0][1, 1, 8, 2][0, 0, 3, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @slice_memcpy(%a : memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd (%a[0, 0, 0, 0][1, 1, 1031, 2][0, 0, 3, 1])
        { metadata = @fifo, id = 0 : i64 } : memref<4096xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @fifo (%tile_0_0, MM2S, 0)
  }
}
