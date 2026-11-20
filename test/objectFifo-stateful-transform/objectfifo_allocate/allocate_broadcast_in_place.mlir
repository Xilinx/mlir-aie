//===- allocate_broadcast_in_place.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split %s | FileCheck %s --check-prefix=SPLIT
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s --check-prefix=LOWER

// A fifo of several consumers allocated on its producer's tile keeps its pool
// there: a consumer whose core reaches that memory (the producer's own, or a
// neighbour's) drains it in place, the rest through the DMA broadcast. Each
// drainer holds a lock pair of its own, a whole-object segment of the pool
// the producer fills every one of, so the producer waits for every reader.

// SPLIT-LABEL: module @broadcast_with_own_core
// SPLIT:       %[[T:.*]] = aie.tile(0, 2)
// SPLIT-NOT:   @b__cons0__dma
// SPLIT-DAG:   aie.objectfifo.pool @b__pool(%[[T]])
// SPLIT-DAG:   aie.objectfifo.core_endpoint @b__prod(%[[T]]) fills @b__pool {segments = [@s0, @s1]}
// SPLIT-DAG:   aie.objectfifo.dma_endpoint @b__prod__dma(%[[T]]) drains @b__pool {fifoName = "b", segments = [@s0]}
// SPLIT-DAG:   aie.objectfifo.core_endpoint @b__cons0(%[[T]]) drains @b__pool {segments = [@s1]}
// SPLIT-DAG:   aie.route from @b__prod__dma to [@b__cons1__dma]
// SPLIT:       aie.core

// LOWER-LABEL: module @broadcast_with_own_core
// LOWER-DAG:   %[[T:.*]] = aie.tile(0, 2)
// LOWER-DAG:   %[[F:.*]] = aie.tile(0, 5)
// LOWER-DAG:   aie.flow(%[[T]], DMA : 0, %[[F]], DMA : 0)
// LOWER-NOT:   aie.flow(%{{.*}}, DMA : {{[0-9]}}, %[[T]], DMA
module @broadcast_with_own_core {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %f = aie.tile(0, 5)
    aie.objectfifo @b(%t, {%t, %f}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @b (%t)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      %1 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.core(%f) {
      %0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

// -----

// A neighbour north of the producer also reaches its memory; only the far
// tile receives by DMA.

// SPLIT-LABEL: module @broadcast_with_a_neighbour
// SPLIT-DAG:   aie.objectfifo.core_endpoint @b__prod(%{{.*}}) fills @b__pool {segments = [@s0, @s1, @s2]}
// SPLIT-DAG:   aie.objectfifo.core_endpoint @b__cons0(%{{.*}}) drains @b__pool {segments = [@s1]}
// SPLIT-DAG:   aie.objectfifo.core_endpoint @b__cons1(%{{.*}}) drains @b__pool {segments = [@s2]}
// SPLIT-DAG:   aie.route from @b__prod__dma to [@b__cons2__dma]
// SPLIT:       aie.core

// LOWER-LABEL: module @broadcast_with_a_neighbour
// LOWER-DAG:   %[[T:.*]] = aie.tile(0, 2)
// LOWER-DAG:   %[[N:.*]] = aie.tile(0, 3)
// LOWER-DAG:   %[[F:.*]] = aie.tile(0, 5)
// LOWER-DAG:   aie.flow(%[[T]], DMA : 0, %[[F]], DMA : 0)
// LOWER-NOT:   aie.flow(%{{.*}}, %[[N]], DMA
module @broadcast_with_a_neighbour {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %n = aie.tile(0, 3)
    %f = aie.tile(0, 5)
    aie.objectfifo @b(%t, {%t, %n, %f}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @b (%t)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      %1 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.core(%n) {
      %0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.core(%f) {
      %0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
