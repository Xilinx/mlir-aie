// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s --check-prefix=DEFAULT
// RUN: aie-opt --aie-objectfifo-split="dma-fence-shared-mem=true" %s | FileCheck %s --check-prefix=VIADMA
// RUN: aie-opt --aie-objectfifo-split="warn-unfenced-shared-mem=true" %s 2>&1 | FileCheck %s --check-prefix=WARN

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A cross-tile core->core lock-only shared-memory objectfifo has no
// write-completion barrier: AIE2P locks are bare counters with no commit/
// store-completion signal, so a producer's stores can be observed after its
// lock release lands (a separate lock arbiter) but before they commit -- the
// lock guard is latency-sensitive (only luckier with more pipeline slack, never
// safe). The only architected barrier is DMA completion. Shared memory is the
// fast DEFAULT; correctness is opt-in. --dma-fence-shared-mem carries
// EVERY such fifo on DMA (acquire count is irrelevant -- single-acquire is only
// luckier, not safe). --warn-unfenced-shared-mem (set under the resident
// ctrl-pkt overlay) warns on each such fifo left on the shared path. A self-loop
// (producer == consumer) reads its own memory and is never touched.

module @xtile {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    // Multi-acquire (stencil) cross-tile fifo.
    aie.objectfifo @stencil_of (%t02, {%t03}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    // Single-acquire cross-tile fifo (also converted / warned -- not exempt).
    aie.objectfifo @plain_of (%t03, {%t04}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // Self-loop fifo: same tile, always stays shared.
    aie.objectfifo @self_of (%t04, {%t04}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core02 = aie.core(%t02) {
      %e = aie.objectfifo.acquire @stencil_of (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @stencil_of (Produce, 1)
      aie.end
    }
    %core03 = aie.core(%t03) {
      %e:2 = aie.objectfifo.acquire @stencil_of (Consume, 2) : memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @stencil_of (Consume, 1)
      %f = aie.objectfifo.acquire @plain_of (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @plain_of (Produce, 1)
      aie.end
    }
    %core04 = aie.core(%t04) {
      %f = aie.objectfifo.acquire @plain_of (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @plain_of (Consume, 1)
      %g = aie.objectfifo.acquire @self_of (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @self_of (Produce, 1)
      %h = aie.objectfifo.acquire @self_of (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @self_of (Consume, 1)
      aie.end
    }
  }
}

// Default: every fifo stays on the shared lock-only path (no DMA, no route).
// DEFAULT-LABEL: @xtile
// DEFAULT:     aie.objectfifo.core_endpoint @stencil_of_prod
// DEFAULT:     aie.objectfifo.core_endpoint @plain_of_prod
// DEFAULT-NOT: aie.objectfifo.dma_endpoint
// DEFAULT-NOT: aie.route

// Opt-in: BOTH cross-tile fifos (multi- AND single-acquire) go via DMA; the
// self-loop stays shared.
// VIADMA-LABEL: @xtile
// VIADMA-DAG:   aie.objectfifo.dma_endpoint @stencil_of_prod_dma
// VIADMA-DAG:   aie.route from @stencil_of_prod_dma to [@stencil_of_cons_dma]
// VIADMA-DAG:   aie.objectfifo.dma_endpoint @plain_of_prod_dma
// VIADMA-DAG:   aie.route from @plain_of_prod_dma to [@plain_of_cons_dma]
// VIADMA:       aie.objectfifo.core_endpoint @self_of_prod
// VIADMA-NOT:   aie.objectfifo.dma_endpoint @self_of

// Warn (overlay): each cross-tile shared fifo left on the shared path warns;
// the self-loop does not.
// WARN: warning: objectfifo 'stencil_of' uses a cross-tile core-to-core lock-only shared-memory path
// WARN: warning: objectfifo 'plain_of' uses a cross-tile core-to-core lock-only shared-memory path
// WARN-NOT: objectfifo 'self_of' uses cross-tile shared memory
