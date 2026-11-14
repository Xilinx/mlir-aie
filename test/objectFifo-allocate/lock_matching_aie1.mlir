//===- lock_matching_aie1.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: aie-opt --aie-objectFifo-stateful-transform %t.mlir | FileCheck %s

// AIE1 pools take one lock an object. With every pool two deep, each takes
// two, and the column's 64 locks hold them all only as a matching, as on
// AIE2 (lock_matching.mlir): placed in pool order, the last find no room.
// CHECK-COUNT-64: aie.lock(

module @aie1_lock_matching {
  aie.device(xcvc1902) {
    %a = aie.tile(5, 3)
    %b = aie.tile(5, 4)
    %c = aie.tile(5, 1)
    %d = aie.tile(5, 2)
    aie.objectfifo @a0(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a1(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a2(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a3(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a4(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a5(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a6(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a7(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a8(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a9(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a10(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a11(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b0(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b2(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b3(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b4(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b5(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b6(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b7(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b8(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b9(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b10(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b11(%b, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c0(%c, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c1(%c, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c2(%c, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c3(%c, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d0(%d, {%d}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d1(%d, {%d}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d2(%d, {%d}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d3(%d, {%d}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
