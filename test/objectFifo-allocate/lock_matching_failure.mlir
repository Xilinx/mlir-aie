//===- lock_matching_failure.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: not aie-opt --aie-objectFifo-stateful-transform %t.mlir 2>&1 | FileCheck %s

// Column 0's segments fit only as a matching, which the in-order pass misses
// first; @z's 18 locks fit nowhere, since no tile next to (2, 2) is placed.
// The error names the group the matching cannot place, on (2, 2).
// CHECK: error: 'aie.tile' op could not place locks within tile lock capacity
// CHECK: note: see current operation: {{.*}}col = 2 : i32, row = 2 : i32

module @lock_report {
  aie.device(npu1) {
    %a = aie.tile(0, 4)
    %b = aie.tile(0, 5)
    %c = aie.tile(0, 2)
    %d = aie.tile(0, 3)
    %z = aie.tile(2, 2)
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
    aie.objectfifo @z0(%z, {%z}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z1(%z, {%z}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z2(%z, {%z}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z3(%z, {%z}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z4(%z, {%z}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z5(%z, {%z}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z6(%z, {%z}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z7(%z, {%z}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z8(%z, {%z}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
