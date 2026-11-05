//===- test_place_peer_dma_reason.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --aie-place-tiles %s

// A core-to-core fifo whose producer reshapes the data moves by DMA wherever
// its ends sit (ObjectFifoCreateOp::dmaReason), so it costs its consumer an
// input channel even beside its producer: with two from the shim, three, one
// more than a core has.
module @peer_reshapes {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    // expected-error@+2 {{requires 3 input/0 output DMA channels, but only 2 input/2 output available}}
    // expected-note@+1 {{placer selected this tile}}
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0 (%shim, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1 (%shim1, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @seed (%shim2, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @peer (%b dimensionsToStream [<size = 4, stride = 4>, <size = 4, stride = 1>], {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
