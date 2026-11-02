// RUN: aie-opt --aie-objectfifo-split --mlir-print-debuginfo %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A link's shared pool holds the objects of every fifo in the link, so it is
// located at the link and at each of those fifos. The link alone usually has
// only a source line, which says nothing of whose objects the pool holds.

module @link_pool_loc {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @in (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>> loc("in"("design.py":10:4))
    aie.objectfifo @out0 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc("out0"("design.py":11:4))
    aie.objectfifo @out1 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<32xi32>> loc("out1"("design.py":12:4))

    aie.objectfifo.link [@in] -> [@out0, @out1] ([][0, 16]) loc("design.py":20:8)
  }
}

// CHECK:     aie.objectfifo.pool @in__cons__pool(
// CHECK-NEXT:  aie.objectfifo.segment @s0
// CHECK-NEXT:  aie.objectfifo.segment @s1
// CHECK-NEXT: {{^ *}}} loc(#[[POOL:loc[0-9]*]])
// CHECK-DAG: #[[LINK:loc[0-9]*]] = loc("design.py":20:8)
// CHECK-DAG: #[[IN:loc[0-9]*]] = loc("in"(
// CHECK-DAG: #[[OUT0:loc[0-9]*]] = loc("out0"(
// CHECK-DAG: #[[OUT1:loc[0-9]*]] = loc("out1"(
// CHECK-DAG: #[[POOL]] = loc(fused[#[[LINK]], #[[IN]], #[[OUT0]], #[[OUT1]]])
