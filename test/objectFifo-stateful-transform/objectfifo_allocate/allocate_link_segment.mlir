//===- allocate_link_segment.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split %s | FileCheck %s --check-prefix=SPLIT
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s --check-prefix=LOWER

// A join's input, or a distribute's output, whose objects are its segment of
// the link's pool shares the pool where its allocate names the link's tile and
// its other end reaches that tile's memory: the link's own core, or a
// neighbour's. Its core fills or drains its segment in place, with no DMA end
// and no route, so the link tile's channels carry only the far participants.

// The link tile's core writes the first half of each object and reads the
// whole joined object; only the far tile's half arrives by DMA.

// SPLIT-LABEL: module @join_into_own_core
// SPLIT:       %[[T:.*]] = aie.tile(0, 2)
// SPLIT-NOT:   @in0__prod__dma
// SPLIT-NOT:   @out__prod__dma
// SPLIT-NOT:   @out__cons__dma
// SPLIT-DAG:   aie.objectfifo.pool @out__pool(%[[T]])
// SPLIT-DAG:   aie.objectfifo.core_endpoint @in0__prod(%[[T]]) fills @out__pool {segments = [@s0]}
// SPLIT-DAG:   aie.objectfifo.dma_endpoint @in1__cons__dma(%[[T]]) fills @out__pool {fifoName = "in1", segments = [@s1]}
// SPLIT-DAG:   aie.route from @in1__prod__dma to [@in1__cons__dma]
// SPLIT-DAG:   aie.objectfifo.core_endpoint @out__cons(%[[T]]) drains @out__pool {segments = [@s0, @s1]}
// SPLIT:       aie.core

// LOWER-LABEL: module @join_into_own_core
// LOWER-DAG:   %[[T:.*]] = aie.tile(0, 2)
// LOWER-DAG:   %[[F:.*]] = aie.tile(0, 5)
// LOWER-DAG:   aie.flow(%[[F]], DMA : 0, %[[T]], DMA : 0)
// LOWER-NOT:   aie.flow(%[[T]], DMA
module @join_into_own_core {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %f = aie.tile(0, 5)
    aie.objectfifo @in0(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%f, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 16] [])
    aie.objectfifo.allocate @in0 (%t)
    aie.objectfifo.allocate @out (%t)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      %1 = aie.objectfifo.acquire @out(Consume, 1) : memref<32xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
    aie.core(%f) {
      %0 = aie.objectfifo.acquire @in1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in1(Produce, 1)
      aie.end
    }
  }
}

// -----

// A neighbour north of the link tile reaches its memory: both halves are
// written in place, and nothing reaches the link tile by DMA.

// SPLIT-LABEL: module @join_from_neighbour
// SPLIT-NOT:   aie.route
// SPLIT-NOT:   dma_endpoint
// SPLIT-DAG:   aie.objectfifo.core_endpoint @in0__prod(%{{.*}}) fills @out__pool {segments = [@s0]}
// SPLIT-DAG:   aie.objectfifo.core_endpoint @in1__prod(%{{.*}}) fills @out__pool {segments = [@s1]}
// SPLIT-DAG:   aie.objectfifo.core_endpoint @out__cons(%{{.*}}) drains @out__pool {segments = [@s0, @s1]}
// SPLIT:       aie.core

// LOWER-LABEL: module @join_from_neighbour
// LOWER-NOT:   aie.flow
// LOWER:       aie.end
module @join_from_neighbour {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %n = aie.tile(0, 3)
    aie.objectfifo @in0(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%n, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 16] [])
    aie.objectfifo.allocate @in0 (%t)
    aie.objectfifo.allocate @in1 (%t)
    aie.objectfifo.allocate @out (%t)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      %1 = aie.objectfifo.acquire @out(Consume, 1) : memref<32xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
    aie.core(%n) {
      %0 = aie.objectfifo.acquire @in1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in1(Produce, 1)
      aie.end
    }
  }
}

// -----

// The link tile's core writes whole objects and reads its own half; the other
// half leaves by DMA for a far tile.

// SPLIT-LABEL: module @distribute_into_own_core
// SPLIT:       %[[T:.*]] = aie.tile(0, 2)
// SPLIT-NOT:   @in__prod__dma
// SPLIT-NOT:   @in__cons__dma
// SPLIT-NOT:   @out0__prod__dma
// SPLIT-DAG:   aie.objectfifo.pool @in__pool(%[[T]])
// SPLIT-DAG:   aie.objectfifo.core_endpoint @in__prod(%[[T]]) fills @in__pool
// SPLIT-DAG:   aie.objectfifo.core_endpoint @out0__cons(%[[T]]) drains @in__pool {segments = [@s0]}
// SPLIT-DAG:   aie.objectfifo.dma_endpoint @out1__prod__dma(%[[T]]) drains @in__pool {fifoName = "out1", segments = [@s1]}
// SPLIT-DAG:   aie.route from @out1__prod__dma to [@out1__cons__dma]
// SPLIT:       aie.core

// LOWER-LABEL: module @distribute_into_own_core
// LOWER-DAG:   %[[T:.*]] = aie.tile(0, 2)
// LOWER-DAG:   %[[F:.*]] = aie.tile(0, 5)
// LOWER-DAG:   aie.flow(%[[T]], DMA : 0, %[[F]], DMA : 0)
// LOWER-NOT:   aie.flow(%{{.*}}, DMA : {{[0-9]}}, %[[T]], DMA
module @distribute_into_own_core {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %f = aie.tile(0, 5)
    aie.objectfifo @in(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out0(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%t, {%f}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out0, @out1] ([] [0, 16])
    aie.objectfifo.allocate @in (%t)
    aie.objectfifo.allocate @out0 (%t)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @in(Produce, 1) : memref<32xi32>
      aie.objectfifo.release @in(Produce, 1)
      %1 = aie.objectfifo.acquire @out0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out0(Consume, 1)
      aie.end
    }
    aie.core(%f) {
      %0 = aie.objectfifo.acquire @out1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out1(Consume, 1)
      aie.end
    }
  }
}
