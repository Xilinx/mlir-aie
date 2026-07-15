//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The per-BD ND access-pattern limit comes from
// AIETargetModel::getBDMaxDims(tileType): MemTile BDs allow 4 dimensions, core
// and shim BDs allow 3. These are the boundary (accepted) cases; the reject
// cases are covered by nd-dma-too-many-dims-{1,2}.mlir.

// RUN: aie-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @memtile_four_dims_ok
module @memtile_four_dims_ok {
  aie.device(xcve2802) {
    %tile = aie.tile(3, 1)
    %buf = aie.buffer(%tile) { sym_name = "buf" } : memref<128xi32>
    %mem = aie.memtile_dma(%tile) {
      %dma = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      // CHECK: aie.dma_bd
      aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 128 sizes = [1, 1, 1, 1] strides = [1, 1, 1, 1])
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: @core_three_dims_ok
module @core_three_dims_ok {
  aie.device(xcve2802) {
    %tile = aie.tile(3, 3)
    %buf = aie.buffer(%tile) { sym_name = "buf" } : memref<128xi32>
    %mem = aie.mem(%tile) {
      %dma = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      // CHECK: aie.dma_bd
      aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 128 sizes = [1, 1, 1] strides = [1, 1, 1])
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
