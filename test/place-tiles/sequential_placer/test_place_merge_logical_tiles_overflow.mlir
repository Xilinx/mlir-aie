//===- test_place_merge_logical_tiles_overflow.mlir ------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// merge-logical-tiles=false errors when there are more LTOs than physical
// non-core tiles can host without sharing. npu1 has 4 ShimNOC cols (0..3);
// 5 unhinted shim LTOs cannot all be placed on distinct shims when merging
// is forbidden.
//
// With merging enabled (default), the same module places successfully —
// LTOs collapse onto fewer physical tiles within the channel budget.

// RUN: not aie-opt --aie-place-tiles='merge-logical-tiles=false' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOMERGE-OOC
// RUN: aie-opt --aie-place-tiles %s | FileCheck %s --check-prefix=MERGE-OK

// NOMERGE-OOC: error: no ShimNOCTile has sufficient DMA capacity for {{[0-9]+ input/[0-9]+ output channels}}
// NOMERGE-OOC: note: to fix, pin this ShimNOCTile

// MERGE-OK-LABEL: @nomerge_overflow
// MERGE-OK:       aie.device(npu1)
// MERGE-OK-NOT:   aie.logical_tile
module @nomerge_overflow {
  aie.device(npu1) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s4 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(1, 2)
    %c2 = aie.tile(2, 2)
    %c3 = aie.tile(3, 2)
    aie.flow(%s0, DMA : 0, %c0, DMA : 0)
    aie.flow(%s1, DMA : 0, %c1, DMA : 0)
    aie.flow(%s2, DMA : 0, %c2, DMA : 0)
    aie.flow(%s3, DMA : 0, %c3, DMA : 0)
    aie.flow(%s4, DMA : 0, %c0, DMA : 1)
  }
}
