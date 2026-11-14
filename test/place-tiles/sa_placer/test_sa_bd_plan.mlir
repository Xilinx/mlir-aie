//===- test_sa_bd_plan.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The SA placer counts a mem tile's BD ids as aie-objectfifo-allocate will
// spend them: each end as the split makes it, channels planned by the ids they
// share, and the deepest pools walked whole until they fit.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s -o /dev/null

// Two links with 20-deep pools: four ends of 20 BD ids, but walked whole,
// one a side, they fit, and the allocator walks them so. No warning.
module @walked {
  aie.device(npu2) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    %t0 = aie.logical_tile<CoreTile>(?, ?)
    %t1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%s, {%m}, [2 : i32, 20 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out0(%m, {%t0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in0] -> [@out0] ([] [])
    aie.objectfifo @in1(%s, {%m}, [2 : i32, 20 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out1(%m, {%t1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in1] -> [@out1] ([] [])
    aie.core(%t0) {
      %o = aie.objectfifo.acquire @out0(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @out0(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %o = aie.objectfifo.acquire @out1(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @out1(Consume, 1)
      aie.end
    }
  }
}

// -----

// A join of three 9-deep inputs: each input end takes 9 ids, the output's 27
// (9 objects of 3 segments), more than a channel half has, and a pool of
// several segments is never walked whole. The placer warns; the allocator
// fails.
// expected-warning@+3 {{SA placer's resource estimate is over capacity}}
// expected-remark@+2 {{could not find a spill-aware allocation}}
module @join {
  aie.device(npu2) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    // expected-note@+1 {{BD ids penalty=120 on (4, 1)}}
    %m = aie.logical_tile<MemTile>(?, ?)
    %t0 = aie.logical_tile<CoreTile>(?, ?)
    %t1 = aie.logical_tile<CoreTile>(?, ?)
    %t2 = aie.logical_tile<CoreTile>(?, ?)
    %t3 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%t0, {%m}, 9 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b(%t1, {%m}, 9 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @c(%t2, {%m}, 9 : i32) : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{needs 27 buffer descriptors}}
    aie.objectfifo @abc(%m, {%t3}, 9 : i32) : !aie.objectfifo<memref<192xi32>>
    aie.objectfifo.link [@a, @b, @c] -> [@abc] ([0, 64, 128] [])
    aie.core(%t0) {
      %o = aie.objectfifo.acquire @a(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    aie.core(%t1) {
      %o = aie.objectfifo.acquire @b(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    aie.core(%t2) {
      %o = aie.objectfifo.acquire @c(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @c(Produce, 1)
      aie.end
    }
    aie.core(%t3) {
      %o = aie.objectfifo.acquire @abc(Consume, 1) : memref<192xi32>
      aie.objectfifo.release @abc(Consume, 1)
      aie.end
    }
  }
}
