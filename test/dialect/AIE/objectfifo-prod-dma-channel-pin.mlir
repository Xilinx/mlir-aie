//===- objectfifo-prod-dma-channel-pin.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// De-risks the Stage-2 enforcement mechanism (union-consistent auto-packetize
// spec 5.4): setting `prod_dma_channel` on a shim objectfifo must land its
// shim MM2S allocation on that exact channel, not wherever first-free
// assignment would otherwise put it. `--aie-objectFifo-stateful-transform`
// runs AIEObjectFifoSplit -> Verify -> AIEObjectFifoAllocate (whose
// assignChannels() honors the pin via reservePinnedChannel(), and whose
// emitShimAllocations() is what actually prints aie.shim_dma_allocation) ->
// LowerDMAs -> LowerCores -> ErasePools, so the allocation is already visible
// straight out of the pipeline; no separate unroll pass is needed.
//
// @pinned pins MM2S channel 1 up front; @free is left unpinned. Pins reserve
// before auto-assignment runs (see
// dma_channel_pinning/pin_reserves_before_autoassign_AIE2.mlir), so @free's
// first-free search skips the reserved channel 1 and lands on channel 0.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" %s | FileCheck %s

// CHECK-DAG: aie.shim_dma_allocation @{{.*}}pinned{{.*}}(%{{.*}}, MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @{{.*}}free{{.*}}(%{{.*}}, MM2S, 0)

module @objectfifo_prod_dma_channel_pin {
  aie.device(npu2) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)

    aie.objectfifo @pinned(%tile00, {%tile02}, 2 : i32) {prod_dma_channel = 1 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @free(%tile00, {%tile03}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
