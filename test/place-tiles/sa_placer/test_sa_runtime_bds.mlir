//===- test_sa_runtime_bds.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Three shim LTOs on two columns. @in_a's tasks hold 12 BDs at once and
// @in_b's 8 (each descriptor splits into 4); together they overflow a shim
// tile's 16, though their consumers pull both to column 0. The SA charges a
// shim position for the runtime BD ids its LTOs' tasks lack, as
// aie-assign-runtime-sequence-bd-ids hands them out, so the two heavy shims
// land on different tiles and the ids fit. Seeds 1, 5 and 7 put both on
// (0, 0) without that term.

// RUN: aie-opt --pass-pipeline='builtin.module(aie-place-tiles{placer=sa_placer sa-seed=1})' %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: aie-opt --pass-pipeline='builtin.module(aie-place-tiles{placer=sa_placer sa-seed=5})' %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie-place-tiles{placer=sa_placer sa-seed=7})' %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' %t | FileCheck %s --check-prefix=IDS
// RUN: aie-opt --pass-pipeline='builtin.module(aie-place-tiles{placer=sequential_placer},aie.device(aie-objectFifo-stateful-transform,aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' %s | FileCheck %s --check-prefix=IDS

// CHECK:     aie.objectfifo @in_a(%[[A:shim_noc_tile_[0-9]+_0]],
// CHECK-NOT: aie.objectfifo @in_b(%[[A]],
// IDS:       aie.runtime_sequence
// IDS:       {bd_id = 11 : i32}

module @heavy_shims_pulled_together {
  aie.device(npu2_2col) {
    %core_a = aie.logical_tile<CoreTile>(0, 2)
    %core_b = aie.logical_tile<CoreTile>(0, 3)
    %core_c = aie.logical_tile<CoreTile>(1, 2)
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_c = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in_a(%shim_a, {%core_a}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo @in_b(%shim_b, {%core_b}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo @in_c(%shim_c, {%core_c}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.core(%core_a) {
      %0 = aie.objectfifo.acquire @in_a(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_a(Consume, 1)
      aie.end
    }
    aie.core(%core_b) {
      %0 = aie.objectfifo.acquire @in_b(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.core(%core_c) {
      %0 = aie.objectfifo.acquire @in_c(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_c(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%a: memref<16777216xbf16>, %b: memref<16777216xbf16>, %c: memref<4096xbf16>) {
      %ta0 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 0 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta0)
      %ta1 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 64 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta1)
      %ta2 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 128 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta2)
      %tb0 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<16777216xbf16> offset = 0 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tb0)
      %tb1 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<16777216xbf16> offset = 64 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tb1)
      %tc0 = aiex.dma_configure_task_for @in_c {
        aie.dma_bd(%c : memref<4096xbf16> offset = 0 len = 4096)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tc0)
      aiex.dma_await_task(%ta0)
      aiex.dma_await_task(%ta1)
      aiex.dma_await_task(%ta2)
      aiex.dma_await_task(%tb0)
      aiex.dma_await_task(%tb1)
      aiex.dma_await_task(%tc0)
    }
  }
}
