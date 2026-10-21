//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The memtile MM2S lock starts at 1, so the channel streams as soon as the
// configuration enables it, before the runtime sequence runs. Every word must
// still reach the host.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)

    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    aie.shim_dma_allocation @out0(%tile_0_0, S2MM, 0)

    aie.runtime_sequence(%arg0: memref<256xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c256 = arith.constant 256 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c256][%c0, %c0, %c0, %c1]) {id = 0 : i64, metadata = @out0, issue_token = true} : memref<256xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }

    %buff = aie.buffer(%tile_0_1) {sym_name = "buff"} : memref<256xi32> = dense<[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]>
    %full = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "full"}
    %empty = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "empty"}

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%full, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buff : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%empty, Release, %c1_i32)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
