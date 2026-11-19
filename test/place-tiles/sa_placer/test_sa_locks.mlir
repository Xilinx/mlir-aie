//===- test_sa_locks.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" --mlir-pass-statistics %t.mlir -o %t 2> %t.err
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
      %x = aie.objectfifo.acquire @ab(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @ab(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %y = aie.objectfifo.acquire @ab(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @ab(Consume, 1)
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
      %x = aie.objectfifo.acquire @ab(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @ab(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %y = aie.objectfifo.acquire @ab(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @ab(Consume, 1)
      aie.end
    }
  }
}
