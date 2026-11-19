//===- test_sa_memtile_ports.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two mem tiles each take three streams from the cores of column 0 above,
// and a mem tile's switchbox takes four from the North. Sharing one mem
// tile fits its six DMA channels, so the SA places them apart only for the
// North ports. Reversed and renamed, the design places alike.

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=2' %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=3' %s | FileCheck %s

// CHECK-LABEL: module @memtile_ports
// CHECK-DAG: aie.tile(0, 1)
// CHECK-DAG: aie.tile(1, 1)
// CHECK-LABEL: module @memtile_ports_permuted
// CHECK-DAG: aie.tile(0, 1)
// CHECK-DAG: aie.tile(1, 1)

module @memtile_ports {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %c2 = aie.logical_tile<CoreTile>(0, 4)
    %c3 = aie.logical_tile<CoreTile>(0, 5)
    %ma = aie.logical_tile<MemTile>(?, ?)
    %mb = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @a0(%c0, {%ma}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a1(%c1, {%ma}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a2(%c2, {%ma}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b0(%c3, {%mb}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%c0, {%mb}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b2(%c1, {%mb}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

module @memtile_ports_permuted {
  aie.device(npu2) {
    %y = aie.logical_tile<MemTile>(?, ?)
    %x = aie.logical_tile<MemTile>(?, ?)
    %k3 = aie.logical_tile<CoreTile>(0, 5)
    %k2 = aie.logical_tile<CoreTile>(0, 4)
    %k1 = aie.logical_tile<CoreTile>(0, 3)
    %k0 = aie.logical_tile<CoreTile>(0, 2)
    aie.objectfifo @q2(%k1, {%y}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q1(%k0, {%y}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q0(%k3, {%y}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p2(%k2, {%x}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p1(%k1, {%x}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p0(%k0, {%x}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
