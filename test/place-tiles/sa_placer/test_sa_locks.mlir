//===- test_sa_locks.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" --mlir-pass-statistics %s -o %t 2> %t.err
// RUN: FileCheck %s --input-file=%t
// RUN: FileCheck %s --check-prefix=STATS --input-file=%t.err
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform))' %t -o /dev/null

// The SA counts the locks aie-objectfifo-allocate can place. @a at (0, 2)
// needs 32 locks and reaches only rows 2 and 3; @b next to it at row 3 needs
// 20 more there or on row 4, 18 short. @ab pulls @b next to @a; with 15
// segments on @a that does not fit, so @b goes to row 4.
// CHECK-LABEL: @lock_short
// CHECK: aie.objectfifo @b0(%tile_0_4

// With 14 segments on @a the locks fit with @b at row 3, which @ab prefers.
// CHECK-LABEL: @lock_fits
// CHECK: aie.objectfifo @b0(%tile_0_3

// The SA counts the lock groups allocation makes: with @a and @b apart, @ab
// has a pool at each end.
// STATS: 28 sa-lock-groups
// STATS-NEXT: 0 sa-lock-penalty
// STATS: 26 sa-lock-groups
// STATS-NEXT: 0 sa-lock-penalty

module @lock_short {
  aie.device(npu1_1col) {
    %a = aie.logical_tile<CoreTile>(0, 2)
    %b = aie.logical_tile<CoreTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %d = aie.logical_tile<CoreTile>(?, ?)
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
    aie.objectfifo @a12(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a13(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a14(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b0(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b2(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b3(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b4(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b5(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b6(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b7(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b8(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c0(%c, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d0(%d, {%d}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab(%a, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
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
      %p12 = aie.objectfifo.acquire @a12(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a12(Produce, 1)
      %c12 = aie.objectfifo.acquire @a12(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a12(Consume, 1)
      %p13 = aie.objectfifo.acquire @a13(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a13(Produce, 1)
      %c13 = aie.objectfifo.acquire @a13(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a13(Consume, 1)
      %p14 = aie.objectfifo.acquire @a14(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a14(Produce, 1)
      %c14 = aie.objectfifo.acquire @a14(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a14(Consume, 1)
      %x = aie.objectfifo.acquire @ab(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @ab(Produce, 1)
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
      %y = aie.objectfifo.acquire @ab(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @ab(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %p0 = aie.objectfifo.acquire @c0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @c0(Produce, 1)
      %c0 = aie.objectfifo.acquire @c0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @c0(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %p0 = aie.objectfifo.acquire @d0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @d0(Produce, 1)
      %c0 = aie.objectfifo.acquire @d0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d0(Consume, 1)
      aie.end
    }
  }
}

// -----

module @lock_fits {
  aie.device(npu1_1col) {
    %a = aie.logical_tile<CoreTile>(0, 2)
    %b = aie.logical_tile<CoreTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %d = aie.logical_tile<CoreTile>(?, ?)
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
    aie.objectfifo @a12(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a13(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b0(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b2(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b3(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b4(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b5(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b6(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b7(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b8(%b, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c0(%c, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d0(%d, {%d}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @ab(%a, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
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
      %p12 = aie.objectfifo.acquire @a12(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a12(Produce, 1)
      %c12 = aie.objectfifo.acquire @a12(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a12(Consume, 1)
      %p13 = aie.objectfifo.acquire @a13(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a13(Produce, 1)
      %c13 = aie.objectfifo.acquire @a13(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a13(Consume, 1)
      %x = aie.objectfifo.acquire @ab(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @ab(Produce, 1)
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
      %y = aie.objectfifo.acquire @ab(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @ab(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %p0 = aie.objectfifo.acquire @c0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @c0(Produce, 1)
      %c0 = aie.objectfifo.acquire @c0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @c0(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %p0 = aie.objectfifo.acquire @d0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @d0(Produce, 1)
      %c0 = aie.objectfifo.acquire @d0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d0(Consume, 1)
      aie.end
    }
  }
}
