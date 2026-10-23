//===- out_of_order_mismatched_pools.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An out-of-order sender names the buffer each object lands in, and one BD
// carries one such name. Ends that hold different buffers cannot be named that
// way: a sender of 2 buffers feeding a receiver of 3 would have to repeat its
// chain 6 times before the pairing came round, and a sender of one large
// buffer would have to be split into one descriptor per receiving buffer.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles --aie-objectfifo-split --aie-objectfifo-verify %s

module @unequal_depth {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    // expected-error @+1 {{sends out of order from}}
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, [2, 3]) {packet, packet_id = 3 : i8, out_of_order, iter_count = 1 : i32} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %1 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %1 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
}

// -----

// One large buffer sent into several smaller ones. The sender has one
// descriptor to name a buffer with and the receiver has four buffers to name.
module @unequal_size {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    // expected-error @+1 {{sends out of order from}}
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, [2, 4]) {packet, packet_id = 3 : i8, out_of_order, iter_count = 1 : i32} : !aie.objectfifo<memref<16xi32>> -> !aie.objectfifo<memref<4xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %1 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %1 = aie.objectfifo.acquire @a(Consume, 1) : memref<4xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
}
