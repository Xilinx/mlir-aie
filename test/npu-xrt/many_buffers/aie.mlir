//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 16-host-buffer stress test: 8 independent S2MM/MM2S passthrough DMA pairs
// across 2 shim columns. Exercises the kernels.json BO-count path end-to-end
// on hardware for both npu1 and npu2.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)

    // Column 0: 4 pairs (MM2S 0,1 + S2MM 0,1)
    aie.shim_dma_allocation @in0  (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @in1  (%tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @out1 (%tile_0_0, S2MM, 1)

    // Column 1: 4 pairs (MM2S 0,1 + S2MM 0,1)
    aie.shim_dma_allocation @in2  (%tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @out2 (%tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @in3  (%tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @out3 (%tile_1_0, S2MM, 1)

    // Column 0: 4 more pairs using MM2S/S2MM 2,3
    aie.shim_dma_allocation @in4  (%tile_0_0, MM2S, 2)
    aie.shim_dma_allocation @out4 (%tile_0_0, S2MM, 2)
    aie.shim_dma_allocation @in5  (%tile_0_0, MM2S, 3)
    aie.shim_dma_allocation @out5 (%tile_0_0, S2MM, 3)

    // Column 1: 4 more pairs using MM2S/S2MM 2,3
    aie.shim_dma_allocation @in6  (%tile_1_0, MM2S, 2)
    aie.shim_dma_allocation @out6 (%tile_1_0, S2MM, 2)
    aie.shim_dma_allocation @in7  (%tile_1_0, MM2S, 3)
    aie.shim_dma_allocation @out7 (%tile_1_0, S2MM, 3)

    aie.runtime_sequence(
        %in0:  memref<64xi32>, %out0: memref<64xi32>,
        %in1:  memref<64xi32>, %out1: memref<64xi32>,
        %in2:  memref<64xi32>, %out2: memref<64xi32>,
        %in3:  memref<64xi32>, %out3: memref<64xi32>,
        %in4:  memref<64xi32>, %out4: memref<64xi32>,
        %in5:  memref<64xi32>, %out5: memref<64xi32>,
        %in6:  memref<64xi32>, %out6: memref<64xi32>,
        %in7:  memref<64xi32>, %out7: memref<64xi32>
    ) {
      aiex.npu.dma_memcpy_nd(%in0[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 0  : i64, metadata = @in0,  issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out0[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 1  : i64, metadata = @out0, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in1[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 2  : i64, metadata = @in1,  issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out1[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 3  : i64, metadata = @out1, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in2[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 4  : i64, metadata = @in2,  issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out2[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 5  : i64, metadata = @out2, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in3[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 6  : i64, metadata = @in3,  issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out3[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 7  : i64, metadata = @out3, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in4[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 8  : i64, metadata = @in4,  issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out4[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 9  : i64, metadata = @out4, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in5[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 10 : i64, metadata = @in5,  issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out5[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 11 : i64, metadata = @out5, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in6[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 12 : i64, metadata = @in6,  issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out6[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 13 : i64, metadata = @out6, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in7[0,0,0,0][1,1,1,64][0,0,0,1])  {id = 14 : i64, metadata = @in7,  issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out7[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 15 : i64, metadata = @out7, issue_token = true} : memref<64xi32>

      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 1 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 2 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 3 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 2 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 3 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
