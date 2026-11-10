//===- lock_matching_failure.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

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
    aie.core(%a) {
      %p0 = aie.objectfifo.acquire @a0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a0(Produce, 1)
      %c0 = aie.objectfifo.acquire @a0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a0(Consume, 1)
      %p1 = aie.objectfifo.acquire @a1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a1(Produce, 1)
      %c1 = aie.objectfifo.acquire @a1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a1(Consume, 1)
      %p2 = aie.objectfifo.acquire @a2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a2(Produce, 1)
      %c2 = aie.objectfifo.acquire @a2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a2(Consume, 1)
      %p3 = aie.objectfifo.acquire @a3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a3(Produce, 1)
      %c3 = aie.objectfifo.acquire @a3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a3(Consume, 1)
      %p4 = aie.objectfifo.acquire @a4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a4(Produce, 1)
      %c4 = aie.objectfifo.acquire @a4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a4(Consume, 1)
      %p5 = aie.objectfifo.acquire @a5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a5(Produce, 1)
      %c5 = aie.objectfifo.acquire @a5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a5(Consume, 1)
      %p6 = aie.objectfifo.acquire @a6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a6(Produce, 1)
      %c6 = aie.objectfifo.acquire @a6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a6(Consume, 1)
      %p7 = aie.objectfifo.acquire @a7(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a7(Produce, 1)
      %c7 = aie.objectfifo.acquire @a7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a7(Consume, 1)
      %p8 = aie.objectfifo.acquire @a8(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a8(Produce, 1)
      %c8 = aie.objectfifo.acquire @a8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a8(Consume, 1)
      %p9 = aie.objectfifo.acquire @a9(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a9(Produce, 1)
      %c9 = aie.objectfifo.acquire @a9(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a9(Consume, 1)
      %p10 = aie.objectfifo.acquire @a10(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a10(Produce, 1)
      %c10 = aie.objectfifo.acquire @a10(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a10(Consume, 1)
      %p11 = aie.objectfifo.acquire @a11(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a11(Produce, 1)
      %c11 = aie.objectfifo.acquire @a11(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a11(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %p0 = aie.objectfifo.acquire @b0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b0(Produce, 1)
      %c0 = aie.objectfifo.acquire @b0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b0(Consume, 1)
      %p1 = aie.objectfifo.acquire @b1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b1(Produce, 1)
      %c1 = aie.objectfifo.acquire @b1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b1(Consume, 1)
      %p2 = aie.objectfifo.acquire @b2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b2(Produce, 1)
      %c2 = aie.objectfifo.acquire @b2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b2(Consume, 1)
      %p3 = aie.objectfifo.acquire @b3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b3(Produce, 1)
      %c3 = aie.objectfifo.acquire @b3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b3(Consume, 1)
      %p4 = aie.objectfifo.acquire @b4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b4(Produce, 1)
      %c4 = aie.objectfifo.acquire @b4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b4(Consume, 1)
      %p5 = aie.objectfifo.acquire @b5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b5(Produce, 1)
      %c5 = aie.objectfifo.acquire @b5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b5(Consume, 1)
      %p6 = aie.objectfifo.acquire @b6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b6(Produce, 1)
      %c6 = aie.objectfifo.acquire @b6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b6(Consume, 1)
      %p7 = aie.objectfifo.acquire @b7(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b7(Produce, 1)
      %c7 = aie.objectfifo.acquire @b7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b7(Consume, 1)
      %p8 = aie.objectfifo.acquire @b8(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b8(Produce, 1)
      %c8 = aie.objectfifo.acquire @b8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b8(Consume, 1)
      %p9 = aie.objectfifo.acquire @b9(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b9(Produce, 1)
      %c9 = aie.objectfifo.acquire @b9(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b9(Consume, 1)
      %p10 = aie.objectfifo.acquire @b10(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b10(Produce, 1)
      %c10 = aie.objectfifo.acquire @b10(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b10(Consume, 1)
      %p11 = aie.objectfifo.acquire @b11(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b11(Produce, 1)
      %c11 = aie.objectfifo.acquire @b11(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b11(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %p0 = aie.objectfifo.acquire @c0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @c0(Produce, 1)
      %c0 = aie.objectfifo.acquire @c0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @c0(Consume, 1)
      %p1 = aie.objectfifo.acquire @c1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @c1(Produce, 1)
      %c1 = aie.objectfifo.acquire @c1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @c1(Consume, 1)
      %p2 = aie.objectfifo.acquire @c2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @c2(Produce, 1)
      %c2 = aie.objectfifo.acquire @c2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @c2(Consume, 1)
      %p3 = aie.objectfifo.acquire @c3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @c3(Produce, 1)
      %c3 = aie.objectfifo.acquire @c3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @c3(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %p0 = aie.objectfifo.acquire @d0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @d0(Produce, 1)
      %c0 = aie.objectfifo.acquire @d0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d0(Consume, 1)
      %p1 = aie.objectfifo.acquire @d1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @d1(Produce, 1)
      %c1 = aie.objectfifo.acquire @d1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d1(Consume, 1)
      %p2 = aie.objectfifo.acquire @d2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @d2(Produce, 1)
      %c2 = aie.objectfifo.acquire @d2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d2(Consume, 1)
      %p3 = aie.objectfifo.acquire @d3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @d3(Produce, 1)
      %c3 = aie.objectfifo.acquire @d3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d3(Consume, 1)
      aie.end
    }
    aie.core(%z) {
      %p0 = aie.objectfifo.acquire @z0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @z0(Produce, 1)
      %c0 = aie.objectfifo.acquire @z0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @z0(Consume, 1)
      %p1 = aie.objectfifo.acquire @z1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @z1(Produce, 1)
      %c1 = aie.objectfifo.acquire @z1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @z1(Consume, 1)
      %p2 = aie.objectfifo.acquire @z2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @z2(Produce, 1)
      %c2 = aie.objectfifo.acquire @z2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @z2(Consume, 1)
      %p3 = aie.objectfifo.acquire @z3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @z3(Produce, 1)
      %c3 = aie.objectfifo.acquire @z3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @z3(Consume, 1)
      %p4 = aie.objectfifo.acquire @z4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @z4(Produce, 1)
      %c4 = aie.objectfifo.acquire @z4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @z4(Consume, 1)
      %p5 = aie.objectfifo.acquire @z5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @z5(Produce, 1)
      %c5 = aie.objectfifo.acquire @z5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @z5(Consume, 1)
      %p6 = aie.objectfifo.acquire @z6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @z6(Produce, 1)
      %c6 = aie.objectfifo.acquire @z6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @z6(Consume, 1)
      %p7 = aie.objectfifo.acquire @z7(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @z7(Produce, 1)
      %c7 = aie.objectfifo.acquire @z7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @z7(Consume, 1)
      %p8 = aie.objectfifo.acquire @z8(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @z8(Produce, 1)
      %c8 = aie.objectfifo.acquire @z8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @z8(Consume, 1)
      aie.end
    }
  }
}
