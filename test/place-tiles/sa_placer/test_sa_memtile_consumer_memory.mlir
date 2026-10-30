//===- test_sa_memtile_consumer_memory.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' %s -o /dev/null

// Neither the consumer's home nor its only neighbor has room for its two
// 16KB objects. A shim producer must not hide this consumer from the estimate.
module @shim_to_memtile {
  // expected-warning@+1 {{SA placer's resource estimate is over capacity (penalty=960)}}
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %mem = aie.logical_tile<MemTile>(0, 1)
    %neighbor = aie.logical_tile<MemTile>(1, 1)
    %full = aie.buffer(%mem) : memref<524288xi8>
    %full_neighbor = aie.buffer(%neighbor) : memref<524288xi8>
    aie.objectfifo @in(%shim, {%mem}, [1, 2]) : !aie.objectfifo<memref<16384xi8>>
  }
}

// -----

// The same accounting applies when a core feeds the memtile.
module @core_to_memtile {
  // expected-warning@+1 {{SA placer's resource estimate is over capacity (penalty=960)}}
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(0, 2)
    %mem = aie.logical_tile<MemTile>(0, 1)
    %neighbor = aie.logical_tile<MemTile>(1, 1)
    %full = aie.buffer(%mem) : memref<524288xi8>
    %full_neighbor = aie.buffer(%neighbor) : memref<524288xi8>
    aie.objectfifo @in(%core, {%mem}, [1, 2]) : !aie.objectfifo<memref<16384xi8>>
  }
}

// -----

// @mid aliases @in's consumer pool only at %src. Its three consumer objects
// still need their own memory at %dst, even though @mid is a linked output.
module @chained_consumer {
  // expected-warning@+1 {{SA placer's resource estimate is over capacity (penalty=1440)}}
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(3, 0)
    %src = aie.logical_tile<MemTile>(3, 1)
    %dst = aie.logical_tile<MemTile>(0, 1)
    %neighbor = aie.logical_tile<MemTile>(1, 1)
    %full = aie.buffer(%dst) : memref<524288xi8>
    %full_neighbor = aie.buffer(%neighbor) : memref<524288xi8>
    aie.objectfifo @in(%shim, {%src}, [1, 2]) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo @mid(%src, {%dst}, [2, 3]) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo @out(%dst, {%shim}, [3, 1]) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo.link [@in] -> [@mid] ([] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
  }
}

// -----

// Each memtile fits exactly: 256KB static plus a 256KB link pool. Counting
// either linked output's producer pool again would cause an unexpected warning.
module @linked_producer_alias {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %src = aie.logical_tile<MemTile>(0, 1)
    %dst = aie.logical_tile<MemTile>(1, 1)
    %neighbor = aie.logical_tile<MemTile>(2, 1)
    %half_src = aie.buffer(%src) : memref<262144xi8>
    %half_dst = aie.buffer(%dst) : memref<262144xi8>
    %full_neighbor = aie.buffer(%neighbor) : memref<524288xi8>
    aie.objectfifo @in(%shim, {%src}, [1, 2]) : !aie.objectfifo<memref<131072xi8>>
    aie.objectfifo @mid(%src, {%dst}, 2 : i32) : !aie.objectfifo<memref<131072xi8>>
    aie.objectfifo @out(%dst, {%shim}, [2, 1]) : !aie.objectfifo<memref<131072xi8>>
    aie.objectfifo.link [@in] -> [@mid] ([] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
  }
}
