//===- good-window-group-disambiguation.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids \
// RUN:         --aie-unroll-runtime-sequence-loops \
// RUN:         --split-input-file %s | FileCheck %s

// Regression: the unroll pass round-robins each rotation window, and the
// counter must be keyed per rotation GROUP, not by window contents alone.
// Distinct groups legitimately carry identical windows -- different tiles
// allocate BD ids independently, and sequential loops on one tile reuse the
// freed pool -- so a contents-only key lets one group's odd copy count bleed
// its rotation offset into the next group, corrupting its bd_ids.

// -----

// Two depth-1 rotations on DIFFERENT tiles, each unrolling to an odd 3 copies
// (prologue + 2 body). Both independently get window [0, 1]. Each group must
// resolve to 0, 1, 0 on its own; the second group must NOT inherit the first
// group's trailing offset (which would flip it to 1, 0, 1).

// CHECK-LABEL: @two_groups_distinct_tiles
// group A on tile(0,0): 0, 1, 0
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// group B on tile(1,0): 0, 1, 0 (independent of group A's parity)
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK-NOT:   bd_id_window
aie.device(npu1) {
  %t00 = aie.tile(0, 0)
  %t10 = aie.tile(1, 0)
  aie.runtime_sequence @two_groups_distinct_tiles(%arg0: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a0 = aiex.dma_configure_task(%t00, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
      aie.end
    }
    aiex.dma_start_task(%a0)
    %alast = scf.for %i = %c1 to %c3 step %c1 iter_args(%prev = %a0) -> (index) {
      %t = aiex.dma_configure_task(%t00, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%alast)
    aiex.dma_free_task(%alast)
    %b0 = aiex.dma_configure_task(%t10, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
      aie.end
    }
    aiex.dma_start_task(%b0)
    %blast = scf.for %j = %c1 to %c3 step %c1 iter_args(%prev = %b0) -> (index) {
      %t = aiex.dma_configure_task(%t10, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%blast)
    aiex.dma_free_task(%blast)
  }
}

// -----

// Two SEQUENTIAL depth-1 rotations on the SAME tile. The first window's ids are
// freed and reused by the second, so both windows are [0, 1] again. Same
// requirement: each group resolves 0, 1, 0 independently.

// CHECK-LABEL: @two_groups_same_tile
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}{bd_id = 0 : i32}
// CHECK-NOT:   bd_id_window
aie.device(npu1) {
  %t00 = aie.tile(0, 0)
  aie.runtime_sequence @two_groups_same_tile(%arg0: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a0 = aiex.dma_configure_task(%t00, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
      aie.end
    }
    aiex.dma_start_task(%a0)
    %alast = scf.for %i = %c1 to %c3 step %c1 iter_args(%prev = %a0) -> (index) {
      %t = aiex.dma_configure_task(%t00, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%alast)
    aiex.dma_free_task(%alast)
    %b0 = aiex.dma_configure_task(%t00, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
      aie.end
    }
    aiex.dma_start_task(%b0)
    %blast = scf.for %j = %c1 to %c3 step %c1 iter_args(%prev = %b0) -> (index) {
      %t = aiex.dma_configure_task(%t00, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 256)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%blast)
    aiex.dma_free_task(%blast)
  }
}

// -----

// Multi-dma_bd chain (C=2) rotated on two different tiles. Within a group the
// two descriptors use disjoint windows [0, 1] and [2, 3] (disambiguated by
// contents); across groups the identical windows are disambiguated by group id.
// Each group must resolve descriptor 0 -> 0, 1, 0 and descriptor 1 -> 2, 3, 2.

// CHECK-LABEL: @chain_c2_two_groups
// group A: (0,2) (1,3) (0,2)
// CHECK:       aie.dma_bd{{.*}}, 0, 4) {bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}, 4, 4) {bd_id = 2 : i32}
// CHECK:       aie.dma_bd{{.*}}, 0, 4) {bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}, 4, 4) {bd_id = 3 : i32}
// CHECK:       aie.dma_bd{{.*}}, 0, 4) {bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}, 4, 4) {bd_id = 2 : i32}
// group B: identical, not offset by group A's parity
// CHECK:       aie.dma_bd{{.*}}, 0, 4) {bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}, 4, 4) {bd_id = 2 : i32}
// CHECK:       aie.dma_bd{{.*}}, 0, 4) {bd_id = 1 : i32}
// CHECK:       aie.dma_bd{{.*}}, 4, 4) {bd_id = 3 : i32}
// CHECK:       aie.dma_bd{{.*}}, 0, 4) {bd_id = 0 : i32}
// CHECK:       aie.dma_bd{{.*}}, 4, 4) {bd_id = 2 : i32}
// CHECK-NOT:   bd_id_window
aie.device(npu2) {
  %t00 = aie.tile(0, 0)
  %t10 = aie.tile(1, 0)
  aie.runtime_sequence @chain_c2_two_groups(%arg0: memref<8xi16>) {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a = aiex.dma_configure_task(%t00, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 4)
      aie.next_bd ^b
    ^b:
      aie.dma_bd(%arg0 : memref<8xi16>, 4, 4)
      aie.end
    }
    aiex.dma_start_task(%a)
    %al = scf.for %i = %c1 to %c3 step %c1 iter_args(%p = %a) -> (index) {
      %t = aiex.dma_configure_task(%t00, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 4)
        aie.next_bd ^b
      ^b:
        aie.dma_bd(%arg0 : memref<8xi16>, 4, 4)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%p)
      scf.yield %t : index
    }
    aiex.dma_free_task(%al)
    %c = aiex.dma_configure_task(%t10, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 4)
      aie.next_bd ^d
    ^d:
      aie.dma_bd(%arg0 : memref<8xi16>, 4, 4)
      aie.end
    }
    aiex.dma_start_task(%c)
    %cl = scf.for %j = %c1 to %c3 step %c1 iter_args(%p = %c) -> (index) {
      %t = aiex.dma_configure_task(%t10, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 4)
        aie.next_bd ^d
      ^d:
        aie.dma_bd(%arg0 : memref<8xi16>, 4, 4)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%p)
      scf.yield %t : index
    }
    aiex.dma_free_task(%cl)
  }
}
