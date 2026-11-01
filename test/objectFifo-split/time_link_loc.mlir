// RUN: aie-opt --split-input-file --aie-objectfifo-split --mlir-print-debuginfo %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A time link's shared merge or dispatch endpoint moves every participant's
// objects, so like the link's pool it is located at the link and each fifo.

// CHECK-LABEL: module @merge
// CHECK:     aie.objectfifo.dma_endpoint @out__merge__dma({{.*}} loc(#[[MERGE:loc[0-9]*]])
// CHECK-DAG: #[[LINK:loc[0-9]*]] = loc("design.py":20:8)
// CHECK-DAG: #[[A:loc[0-9]*]] = loc("iron:from_a"(
// CHECK-DAG: #[[B:loc[0-9]*]] = loc("iron:from_b"(
// CHECK-DAG: #[[OUT:loc[0-9]*]] = loc("iron:out"(
// CHECK-DAG: #[[MERGE]] = loc(fused[#[[LINK]], #[[A]], #[[B]], #[[OUT]]])
module @merge {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @from_a(%a, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc("iron:from_a"("design.py":10:4))
    aie.objectfifo @from_b(%b, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc("iron:from_b"("design.py":11:4))
    aie.objectfifo @out(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc("iron:out"("design.py":12:4))
    aie.objectfifo.link [@from_a, @from_b] -> [@out] ([] []) {mode = #aie.link_mode<time>} loc("design.py":20:8)
  }
}

// -----

// CHECK-LABEL: module @dispatch
// CHECK:     aie.objectfifo.dma_endpoint @in__dispatch__dma({{.*}} loc(#[[DISPATCH:loc[0-9]*]])
// CHECK-DAG: #[[LINK:loc[0-9]*]] = loc("design.py":30:8)
// CHECK-DAG: #[[IN:loc[0-9]*]] = loc("iron:in"(
// CHECK-DAG: #[[TA:loc[0-9]*]] = loc("iron:to_a"(
// CHECK-DAG: #[[TB:loc[0-9]*]] = loc("iron:to_b"(
// CHECK-DAG: #[[DISPATCH]] = loc(fused[#[[LINK]], #[[IN]], #[[TA]], #[[TB]]])
module @dispatch {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc("iron:in"("design.py":24:4))
    aie.objectfifo @to_a(%m, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc("iron:to_a"("design.py":25:4))
    aie.objectfifo @to_b(%m, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc("iron:to_b"("design.py":26:4))
    aie.objectfifo.link [@in] -> [@to_a, @to_b] ([] []) {mode = #aie.link_mode<time>} loc("design.py":30:8)
  }
}
