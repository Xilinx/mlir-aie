//===- test_sa_memtile_rows.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-assign-buffer-addresses))' %t -o /dev/null

// VE2802 has mem tiles in rows 1 and 2. The SA keeps each mem tile LTO where
// it put it and budgets every mem tile on its own, spilling as objectFIFO
// allocation does: same row, one column over, where home reaches.


// Pinned to (0, 1) and (0, 2), the two mem tiles stay apart.
// CHECK-LABEL: @memtile_rows
// CHECK:       aie.objectfifo @in(%{{.*}}, {%mem_tile_0_1}
// CHECK:       aie.objectfifo @mid(%mem_tile_0_1, {%mem_tile_0_2}
// CHECK:       aie.objectfifo @out(%mem_tile_0_2,

// Four mem tiles of 320 KB each fit their own 512 KB, though each column
// holds 640 KB.
// CHECK-LABEL: @memtile_rows_budget
// CHECK:       aie.objectfifo @f1(%mem_tile_0_1, {%mem_tile_0_2}
// CHECK:       aie.objectfifo @f2(%mem_tile_0_2, {%mem_tile_1_1}
// CHECK:       aie.objectfifo @f3(%mem_tile_1_1, {%mem_tile_1_2}

module @memtile_rows {
  aie.device(xcve2802) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    %m1 = aie.logical_tile<MemTile>(0, 1)
    %m2 = aie.logical_tile<MemTile>(0, 2)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in(%shim, {%m1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @mid(%m1, {%m2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out(%m2, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@mid] ([] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
  }
}

// -----

module @memtile_rows_budget {
  aie.device(xcve2802) {
    %sink = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(0, 2)
    %m2 = aie.logical_tile<MemTile>(1, 1)
    %m3 = aie.logical_tile<MemTile>(1, 2)
    aie.objectfifo @f0(%shim, {%m0}, 5 : i32) : !aie.objectfifo<memref<16000xi32>>
    aie.objectfifo @f1(%m0, {%m1}, 5 : i32) : !aie.objectfifo<memref<16000xi32>>
    aie.objectfifo @f2(%m1, {%m2}, 5 : i32) : !aie.objectfifo<memref<16000xi32>>
    aie.objectfifo @f3(%m2, {%m3}, 5 : i32) : !aie.objectfifo<memref<16000xi32>>
    aie.objectfifo @f4(%m3, {%sink}, 5 : i32) : !aie.objectfifo<memref<16000xi32>>
    aie.objectfifo.link [@f0] -> [@f1] ([] [])
    aie.objectfifo.link [@f1] -> [@f2] ([] [])
    aie.objectfifo.link [@f2] -> [@f3] ([] [])
    aie.objectfifo.link [@f3] -> [@f4] ([] [])
  }
}
