//===- test_sa_lock_spill.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// The mem tile holds 63 of its 64 locks and so big a buffer that @in's
// second object spills to a neighbour. Locks go only on tiles that exist
// before the spill, so @in's pair finds no room, as without the big buffer;
// the SA charges the same, which sa_lock_check.test checks.
// CHECK: warning: SA placer's resource estimate is over capacity
// CHECK: error: 'aie.tile' op could not place locks within tile lock capacity

module @lock_spill {
  aie.device(npu2) {
    %s = aie.tile(1, 0)
    %m = aie.tile(1, 1)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %l0 = aie.lock(%m, 0)
    %l1 = aie.lock(%m, 1)
    %l2 = aie.lock(%m, 2)
    %l3 = aie.lock(%m, 3)
    %l4 = aie.lock(%m, 4)
    %l5 = aie.lock(%m, 5)
    %l6 = aie.lock(%m, 6)
    %l7 = aie.lock(%m, 7)
    %l8 = aie.lock(%m, 8)
    %l9 = aie.lock(%m, 9)
    %l10 = aie.lock(%m, 10)
    %l11 = aie.lock(%m, 11)
    %l12 = aie.lock(%m, 12)
    %l13 = aie.lock(%m, 13)
    %l14 = aie.lock(%m, 14)
    %l15 = aie.lock(%m, 15)
    %l16 = aie.lock(%m, 16)
    %l17 = aie.lock(%m, 17)
    %l18 = aie.lock(%m, 18)
    %l19 = aie.lock(%m, 19)
    %l20 = aie.lock(%m, 20)
    %l21 = aie.lock(%m, 21)
    %l22 = aie.lock(%m, 22)
    %l23 = aie.lock(%m, 23)
    %l24 = aie.lock(%m, 24)
    %l25 = aie.lock(%m, 25)
    %l26 = aie.lock(%m, 26)
    %l27 = aie.lock(%m, 27)
    %l28 = aie.lock(%m, 28)
    %l29 = aie.lock(%m, 29)
    %l30 = aie.lock(%m, 30)
    %l31 = aie.lock(%m, 31)
    %l32 = aie.lock(%m, 32)
    %l33 = aie.lock(%m, 33)
    %l34 = aie.lock(%m, 34)
    %l35 = aie.lock(%m, 35)
    %l36 = aie.lock(%m, 36)
    %l37 = aie.lock(%m, 37)
    %l38 = aie.lock(%m, 38)
    %l39 = aie.lock(%m, 39)
    %l40 = aie.lock(%m, 40)
    %l41 = aie.lock(%m, 41)
    %l42 = aie.lock(%m, 42)
    %l43 = aie.lock(%m, 43)
    %l44 = aie.lock(%m, 44)
    %l45 = aie.lock(%m, 45)
    %l46 = aie.lock(%m, 46)
    %l47 = aie.lock(%m, 47)
    %l48 = aie.lock(%m, 48)
    %l49 = aie.lock(%m, 49)
    %l50 = aie.lock(%m, 50)
    %l51 = aie.lock(%m, 51)
    %l52 = aie.lock(%m, 52)
    %l53 = aie.lock(%m, 53)
    %l54 = aie.lock(%m, 54)
    %l55 = aie.lock(%m, 55)
    %l56 = aie.lock(%m, 56)
    %l57 = aie.lock(%m, 57)
    %l58 = aie.lock(%m, 58)
    %l59 = aie.lock(%m, 59)
    %l60 = aie.lock(%m, 60)
    %l61 = aie.lock(%m, 61)
    %l62 = aie.lock(%m, 62)
    %big = aie.buffer(%m) {sym_name = "big"} : memref<126000xi32>
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @out(%m, {%c}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<4096xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
  }
}
