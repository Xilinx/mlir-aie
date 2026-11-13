//===- test_sa_core_link_merge.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t 2>&1 | FileCheck %s --allow-empty
// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %t -o /dev/null

// A merge keeps one pool on its tile, the output's two 4 KB objects, beside
// the 50000-byte buffer. Every input consumed there shares it, though only the
// first plans a DMA end, so the SA classifies an end by its link on the tile.
// CHECK-NOT: over capacity

module @merge_core {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @in1(%b, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([] []) {mode = #aie.link_mode<time>}
    %big = aie.buffer(%l) {sym_name = "big"} : memref<12500xi32>
    aie.core(%a) {
      %x12442 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %x9299 = aie.objectfifo.acquire @in1(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in1(Produce, 1)
      aie.end
    }
  }
}
