//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 16-host-buffer stress test for the kernels.json BO-count path.
//
// The runtime_sequence takes 16 host buffers (8 in/out passthrough pairs), so
// XRT must declare bo0..bo15. The data path reuses only 4 physical passthrough
// lanes (2 per shim column, each a core-free shim->memtile->shim objectfifo
// link) across 2 rounds, so it stays within a shim's MM2S 0/1 + S2MM 0/1
// channel budget while still exercising all 16 host buffers end-to-end.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_0 = aie.tile(1, 0)
    %tile_1_1 = aie.tile(1, 1)

    // Column 0: 2 lanes (shim MM2S 0/1 -> memtile -> shim S2MM 0/1).
    aie.objectfifo @inA (%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @outA(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@inA] -> [@outA] ([] [])
    aie.objectfifo @inB (%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @outB(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@inB] -> [@outB] ([] [])

    // Column 1: 2 lanes.
    aie.objectfifo @inC (%tile_1_0, {%tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @outC(%tile_1_1, {%tile_1_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@inC] -> [@outC] ([] [])
    aie.objectfifo @inD (%tile_1_0, {%tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @outD(%tile_1_1, {%tile_1_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@inD] -> [@outD] ([] [])

    aie.runtime_sequence(
        %in0: memref<64xi32>, %out0: memref<64xi32>,
        %in1: memref<64xi32>, %out1: memref<64xi32>,
        %in2: memref<64xi32>, %out2: memref<64xi32>,
        %in3: memref<64xi32>, %out3: memref<64xi32>,
        %in4: memref<64xi32>, %out4: memref<64xi32>,
        %in5: memref<64xi32>, %out5: memref<64xi32>,
        %in6: memref<64xi32>, %out6: memref<64xi32>,
        %in7: memref<64xi32>, %out7: memref<64xi32>
    ) {
      // Round 1: pairs 0..3 over lanes A,B,C,D.
      aiex.npu.dma_memcpy_nd(%in0[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 0 : i64, metadata = @inA}  : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out0[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 1 : i64, metadata = @outA, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in1[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 2 : i64, metadata = @inB}  : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out1[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 3 : i64, metadata = @outB, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in2[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 4 : i64, metadata = @inC}  : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out2[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 5 : i64, metadata = @outC, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in3[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 6 : i64, metadata = @inD}  : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out3[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 7 : i64, metadata = @outD, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait { symbol = @outA }
      aiex.npu.dma_wait { symbol = @outB }
      aiex.npu.dma_wait { symbol = @outC }
      aiex.npu.dma_wait { symbol = @outD }

      // Round 2: pairs 4..7 reuse the same 4 lanes with different host buffers.
      aiex.npu.dma_memcpy_nd(%in4[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 8  : i64, metadata = @inA}  : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out4[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 9  : i64, metadata = @outA, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in5[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 10 : i64, metadata = @inB}  : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out5[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 11 : i64, metadata = @outB, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in6[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 12 : i64, metadata = @inC}  : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out6[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 13 : i64, metadata = @outC, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in7[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 14 : i64, metadata = @inD}  : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out7[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 15 : i64, metadata = @outD, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait { symbol = @outA }
      aiex.npu.dma_wait { symbol = @outB }
      aiex.npu.dma_wait { symbol = @outC }
      aiex.npu.dma_wait { symbol = @outD }
    }
  }
}
