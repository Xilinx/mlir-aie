//===- axcache_lowering.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A configured $axcache must reach shim BD word[5] bits [27:24]. These check the
// packed word rather than the attribute so the bit position is pinned, not just
// the plumbing. Each case checks the blockwrite global that precedes its
// runtime sequence.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-dma-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s

// axcache = 5 -> 5 << 24 == 0x05000000 == 83886080.
// CHECK: dense<[64, 0, 0, 0, -2147483648, 83886080, 0, 33554432]>
// CHECK: aie.runtime_sequence @static_custom
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @static_custom(%in: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @a, axcache = 5 : i64} : memref<64xi32>
    }
  }
}

// -----

// Unset axcache falls back to AIETargetModel::getDefaultAxCache() == 2, so
// word[5] is 2 << 24 == 0x02000000 == 33554432.
// CHECK: dense<[64, 0, 0, 0, -2147483648, 33554432, 0, 33554432]>
// CHECK: aie.runtime_sequence @static_default
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @static_default(%in: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @a} : memref<64xi32>
    }
  }
}

// -----

// A runtime size on a strided (ND) transfer takes the dynamic encoder, which
// rebuilds word[5] as an SSA or-tree seeded with the AxCACHE constant. The
// configured value must survive that path too, not just the blockwrite
// template.
// CHECK: aie.runtime_sequence @dynamic_custom
// CHECK: %[[AXC:.*]] = arith.constant 83886080 : i32
// CHECK: arith.ori %[[AXC]]
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @dynamic_custom(%in: memref<4096xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,%n,64][0,0,128,1])
        {id = 0 : i64, metadata = @a, axcache = 5 : i64} : memref<4096xi32>
    }
  }
}
