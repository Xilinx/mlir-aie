//===- test_sa_channel_demand.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The SA placer charges a DMA channel to each end the split gives its own
// (ObjectFifoCreateOp::plannedDmaEnd), so a merge's inputs share the one
// channel they take turns on: seven of them fit a mem tile's six input
// channels, and place and allocate without a warning.

// RUN: aie-opt --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s -o /dev/null

module {
  aie.device(npu2) {
    %m = aie.logical_tile<MemTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %p0 = aie.logical_tile<CoreTile>(?, ?)
    %p1 = aie.logical_tile<CoreTile>(?, ?)
    %p2 = aie.logical_tile<CoreTile>(?, ?)
    %p3 = aie.logical_tile<CoreTile>(?, ?)
    %p4 = aie.logical_tile<CoreTile>(?, ?)
    %p5 = aie.logical_tile<CoreTile>(?, ?)
    %p6 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%p0, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @in1(%p1, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @in2(%p2, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @in3(%p3, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @in4(%p4, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @in5(%p5, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @in6(%p6, {%m}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out(%m, {%c}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in0, @in1, @in2, @in3, @in4, @in5, @in6] -> [@out] ([] []) {mode = #aie.link_mode<time>}
    aie.core(%p0) {
      %x = aie.objectfifo.acquire @in0(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%p1) {
      %x = aie.objectfifo.acquire @in1(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @in1(Produce, 1)
      aie.end
    }
    aie.core(%p2) {
      %x = aie.objectfifo.acquire @in2(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @in2(Produce, 1)
      aie.end
    }
    aie.core(%p3) {
      %x = aie.objectfifo.acquire @in3(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @in3(Produce, 1)
      aie.end
    }
    aie.core(%p4) {
      %x = aie.objectfifo.acquire @in4(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @in4(Produce, 1)
      aie.end
    }
    aie.core(%p5) {
      %x = aie.objectfifo.acquire @in5(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @in5(Produce, 1)
      aie.end
    }
    aie.core(%p6) {
      %x = aie.objectfifo.acquire @in6(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @in6(Produce, 1)
      aie.end
    }
    aie.core(%c) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c7 = arith.constant 7 : index
      scf.for %i = %c0 to %c7 step %c1 {
        %x = aie.objectfifo.acquire @out(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @out(Consume, 1)
      }
      aie.end
    }
  }
}
