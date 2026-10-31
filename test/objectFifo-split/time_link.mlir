//===- time_link.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split %s | FileCheck %s

// A dispatch shares one draining endpoint on the link point: it takes turns
// between the outputs' consumers in the link's output order, over one
// packet-switched route each. A pinned header is kept; the others are open.
// CHECK-LABEL: module @dispatch
// CHECK:       aie.objectfifo.pool @in__cons__pool(%[[MEM:.*]]) {depth = 2 : i32
// CHECK:       aie.objectfifo.dma_endpoint @in__cons__dma(%[[MEM]]) fills @in__cons__pool
// CHECK:       aie.objectfifo.dma_endpoint @in__dispatch__dma(%[[MEM]]) drains @in__cons__pool {dispatch = [@to_b__cons__dma, @to_a__cons__dma]
// CHECK:       aie.route from @in__dispatch__dma to [@to_a__cons__dma] {packet = #aie.packet_info<>}
// CHECK:       aie.route from @in__dispatch__dma to [@to_b__cons__dma] {packet = #aie.packet_info<pkt_id = 4>}
// CHECK-NOT:   @to_a__prod__dma
// CHECK-NOT:   @to_b__prod__dma
module @dispatch {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_a(%m, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_b(%m, {%b}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@to_b, @to_a] ([] []) {mode = #aie.link_mode<time>}
  }
}

// -----

// A merge is one fan-in route from the inputs' producers into one filling
// endpoint on the link point.
// CHECK-LABEL: module @merge
// CHECK:       aie.objectfifo.dma_endpoint @from_a__prod__dma
// CHECK:       aie.objectfifo.dma_endpoint @out__merge__dma(%[[MEM:.*]]) fills @out__pool
// CHECK:       aie.objectfifo.dma_endpoint @from_b__prod__dma
// CHECK:       aie.objectfifo.pool @out__pool(%[[MEM]]) {depth = 2 : i32
// CHECK:       aie.route from [@from_a__prod__dma, @from_b__prod__dma] to [@out__merge__dma] {packet = #aie.packet_info<>}
// CHECK-NOT:   @from_a__cons__dma
// CHECK-NOT:   @from_b__cons__dma
module @merge {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @from_a(%a, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @from_b(%b, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@from_a, @from_b] -> [@out] ([] []) {mode = #aie.link_mode<time>}
  }
}
