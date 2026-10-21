//===- test_place_structural_order.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// One design written twice: the second declares its tiles, fifos, links and
// cores in another order and swaps the lanes' names. The lanes differ only in
// the depth of the fifo feeding them, and each lands on the same core in both.

// CHECK-LABEL: module @declared
// CHECK-DAG:   aie.objectfifo @a_mid(%{{.*}}, {%[[A:tile_0_[0-9]]]}, 2 : i32)
// CHECK-DAG:   aie.objectfifo @b_mid(%{{.*}}, {%[[B:tile_0_[0-9]]]}, 3 : i32)
// CHECK-DAG:   aie.core(%[[A]]) {
// CHECK-DAG:   aie.core(%[[B]]) {

// CHECK-LABEL: module @permuted
// CHECK-DAG:   aie.objectfifo @a_mid(%{{.*}}, {%[[B]]}, 3 : i32)
// CHECK-DAG:   aie.objectfifo @b_mid(%{{.*}}, {%[[A]]}, 2 : i32)

module @declared {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core_a = aie.logical_tile<CoreTile>(?, ?)
    %core_b = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a_in(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a_mid(%mem, {%core_a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_in] -> [@a_mid]([] [])
    aie.objectfifo @b_in(%shim, {%mem}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b_mid(%mem, {%core_b}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_in] -> [@b_mid]([] [])
    %ca = aie.core(%core_a) {
      %0 = aie.objectfifo.acquire @a_mid(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a_mid(Consume, 1)
      aie.end
    }
    %cb = aie.core(%core_b) {
      %0 = aie.objectfifo.acquire @b_mid(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b_mid(Consume, 1)
      aie.end
    }
  }
}

// -----

module @permuted {
  aie.device(npu2) {
    %core_a = aie.logical_tile<CoreTile>(?, ?)
    %core_b = aie.logical_tile<CoreTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %ca = aie.core(%core_a) {
      %0 = aie.objectfifo.acquire @a_mid(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a_mid(Consume, 1)
      aie.end
    }
    %cb = aie.core(%core_b) {
      %0 = aie.objectfifo.acquire @b_mid(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b_mid(Consume, 1)
      aie.end
    }
    aie.objectfifo @a_in(%shim, {%mem}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a_mid(%mem, {%core_a}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b_in(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b_mid(%mem, {%core_b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_in] -> [@b_mid]([] [])
    aie.objectfifo.link [@a_in] -> [@a_mid]([] [])
  }
}
