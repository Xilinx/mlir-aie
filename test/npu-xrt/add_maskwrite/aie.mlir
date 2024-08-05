//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    memref.global "public" @out0 : memref<64xi32>

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %input_lock0 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "input_lock0"}
    %input_lock1 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "input_lock1"}

    %output_lock0 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "output_lock0"}
    %output_lock1 = aie.lock(%tile_0_2, 3) {init = 1 : i32, sym_name = "output_lock1"}

    %input_buffer = aie.buffer(%tile_0_2) {sym_name = "input_buffer"} : memref<8xi32>
    %output_buffer = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<8xi32>

    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 926365495 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
            memref.store %c3_i32, %input_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_lock0, AcquireGreaterEqual, 1)
        aie.use_lock(%output_lock1, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
            %2 = arith.addi %1, %c1_i32 : i32
            memref.store %2, %output_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%output_lock0, Release, 1)
        aie.use_lock(%input_lock1, Release, 1)
      }
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%output_lock0, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_buffer : memref<8xi32>) { len = 8 : i32 }
      aie.use_lock(%output_lock1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @out0(S2MM, 0, 0)

    aiex.runtime_sequence @seq(%arg0: memref<8xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c8_i64 = arith.constant 8 : i64

      aiex.npu.maskwrite32 {row = 2 : i32, column = 0 : i32, address = 1024 : ui32, value = 0x12345678 : ui32, mask = 0xF0F0F0F0 : ui32}
      aiex.npu.maskwrite32 {buffer = @input_buffer, address = 1 : ui32, value = 0x9ABCDEF0 : ui32, mask = 0x0F0F0F0F : ui32}

      aiex.npu.dma_memcpy_nd(0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<8xi32>
      aiex.npu.write32 { row = 2 : i32, column = 0 : i32, address = 0x0001F000 : ui32, value = 1 : ui32 }
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
