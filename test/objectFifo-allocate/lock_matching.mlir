//===- lock_matching.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: aie-opt --aie-objectFifo-stateful-transform %t.mlir | FileCheck %s

// Each segment takes two locks on a tile all its users reach. The cores at
// rows 4 and 5 need 48 locks and together reach only rows 3 to 5, which hold
// 48, so the cores at rows 2 and 3 must put all 16 of theirs on row 2.
// Placing segments in pool order puts some of row 3's on row 3 itself;
// assigning them as a matching finds room for all.
// CHECK-COUNT-64: aie.lock(

module @lock_matching {
  aie.device(npu1_1col) {
    %a = aie.tile(0, 4)
    %b = aie.tile(0, 5)
    %c = aie.tile(0, 2)
    %d = aie.tile(0, 3)
    aie.objectfifo @a0(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a1(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a2(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a3(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a4(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a5(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a6(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a7(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a8(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a9(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a10(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a11(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b0(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b2(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b3(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b4(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b5(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b6(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b7(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b8(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b9(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b10(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b11(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c0(%c, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c1(%c, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c2(%c, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c3(%c, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d0(%d, {%d}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d1(%d, {%d}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d2(%d, {%d}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d3(%d, {%d}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
