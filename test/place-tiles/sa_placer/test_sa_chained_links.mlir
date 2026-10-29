//===- test_sa_chained_links.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' --split-input-file %s | FileCheck %s

// The links fit on one memtile, but a chain cannot loop back into its source.
// Verification after placement checks that the two logical tiles stay apart.
// CHECK-LABEL: module @chained_links
// CHECK-NOT: aie.logical_tile
// CHECK: aie.objectfifo @mid
module @chained_links {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %join = aie.logical_tile<MemTile>(?, ?)
    %fwd = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a (%c0, {%join}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b (%c1, {%join}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid (%join, {%fwd}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out (%fwd, {%shim}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@mid] ([0, 16] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
  }
}

// -----

// Both readers of a multicast chain must stay off its producing link's tile.
// CHECK-LABEL: module @multicast_chain
// CHECK-NOT: aie.logical_tile
// CHECK: aie.objectfifo @mid
module @multicast_chain {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %src = aie.logical_tile<MemTile>(?, ?)
    %m2 = aie.logical_tile<MemTile>(?, ?)
    %m3 = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in (%shim, {%src}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @mid (%src, {%m2, %m3}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @x (%m2, {%c0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @y (%m3, {%c1}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@in] -> [@mid] ([] [])
    aie.objectfifo.link [@mid] -> [@x] ([] [])
    aie.objectfifo.link [@mid] -> [@y] ([] [])
  }
}

// -----

// Separation also applies to fixed physical link tiles, in either direction.
// CHECK-LABEL: module @chain_from_physical
// CHECK-NOT: aie.logical_tile
// CHECK: aie.objectfifo @mid
module @chain_from_physical {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %src = aie.tile(0, 1)
    %dst = aie.logical_tile<MemTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in (%shim, {%src}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid (%src, {%dst}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out (%dst, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@mid] ([] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
  }
}

// -----

// CHECK-LABEL: module @chain_into_physical
// CHECK-NOT: aie.logical_tile
// CHECK: aie.objectfifo @mid
module @chain_into_physical {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %src = aie.logical_tile<MemTile>(?, ?)
    %dst = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%src}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid (%src, {%dst}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out (%dst, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@mid] -> [@out] ([] [])
    aie.objectfifo.link [@in] -> [@mid] ([] [])
  }
}
