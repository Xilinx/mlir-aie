//===- test_place_shim_bd_ids.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s -o %t.seq
// RUN: FileCheck %s --input-file=%t.seq
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-assign-bd-ids,aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' %t.seq -o /dev/null
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t.sa
// RUN: FileCheck %s --check-prefix=SA --input-file=%t.sa
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-assign-bd-ids,aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' %t.sa -o /dev/null

// The placers count a shim's BD ids as aie-assign-bd-ids and
// aie-assign-runtime-sequence-bd-ids hand them out, static aie.shim_dma BDs
// and shims placed by hand included.

// @static_shim_bds: @shim_s's static program holds 10 BD ids for good and
// @in_b's tasks 8 at once (each descriptor splits into 4); both are pulled
// to column 0, where together they need 18 of 16.
// CHECK-LABEL: @static_shim_bds
// CHECK:       aie.shim_dma(%[[S:shim_noc_tile_[0-9]+_0]])
// CHECK-NOT:   aie.objectfifo @in_b(%[[S]],
// SA-LABEL:    @static_shim_bds
// SA:          aie.shim_dma(%[[S:shim_noc_tile_[0-9]+_0]])
// SA-NOT:      aie.objectfifo @in_b(%[[S]],

// @placed_shim: aie.tile(0, 0)'s @in_a tasks hold 12 BD ids at once; the
// logical shim's 8 do not fit beside them.
// CHECK-LABEL: @placed_shim
// CHECK:       aie.objectfifo @in_b(%shim_noc_tile_1_0,
// SA-LABEL:    @placed_shim
// SA:          aie.objectfifo @in_b(%shim_noc_tile_1_0,

module @static_shim_bds {
  aie.device(npu2_2col) {
    %core = aie.logical_tile<CoreTile>(0, 2)
    %shim_s = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    %ext = aie.external_buffer {sym_name = "ext"} : memref<1024xi32>
    aie.shim_dma(%shim_s) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%ext : memref<1024xi32> offset = 0 len = 64)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%ext : memref<1024xi32> offset = 64 len = 64)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%ext : memref<1024xi32> offset = 128 len = 64)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%ext : memref<1024xi32> offset = 192 len = 64)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%ext : memref<1024xi32> offset = 256 len = 64)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%ext : memref<1024xi32> offset = 320 len = 64)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%ext : memref<1024xi32> offset = 384 len = 64)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%ext : memref<1024xi32> offset = 448 len = 64)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%ext : memref<1024xi32> offset = 512 len = 64)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%ext : memref<1024xi32> offset = 576 len = 64)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.flow(%shim_s, DMA : 0, %core, DMA : 1)
    aie.objectfifo @in_b(%shim_b, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @in_b(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%b: memref<16777216xbf16>) {
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
      aiex.dma_await_task(%tb0)
      aiex.dma_await_task(%tb1)
    }
  }
}

// -----

module @placed_shim {
  aie.device(npu2_2col) {
    %core_a = aie.logical_tile<CoreTile>(0, 2)
    %core_b = aie.logical_tile<CoreTile>(0, 3)
    %shim_a = aie.tile(0, 0)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in_a(%shim_a, {%core_a}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo @in_b(%shim_b, {%core_b}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
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
    aie.runtime_sequence(%a: memref<16777216xbf16>, %b: memref<16777216xbf16>) {
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
      aiex.dma_await_task(%ta0)
      aiex.dma_await_task(%ta1)
      aiex.dma_await_task(%ta2)
      aiex.dma_await_task(%tb0)
      aiex.dma_await_task(%tb1)
    }
  }
}
