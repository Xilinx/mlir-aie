//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license && jackl
// RUN: xchesscc -p me -P %aietools/data/versal_prod/lib -c %S/kernel.cc %S/dequant.cc %S/pass.cc
// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf



module @idct {
  %t74 = aie.tile(7, 4)
  %t75 = aie.tile(7, 5)

  %t73 = aie.tile(7, 3)
  %t72 = aie.tile(7, 2)
  %t71 = aie.tile(7, 1)
  %t70 = aie.tile(7, 0)

  %buf_73_aping = aie.buffer(%t73) {sym_name = "a73_ping" } : memref<64xi16>
  %buf_73_apong = aie.buffer(%t73) {sym_name = "a73_pong" } : memref<64xi16>
  %buf_73_bping = aie.buffer(%t73) {sym_name = "b73_ping" } : memref<64xi16>
  %buf_73_bpong = aie.buffer(%t73) {sym_name = "b73_pong" } : memref<64xi16>

  %lock_73_a_ping = aie.lock(%t73, 3) // a_ping
  %lock_73_a_pong = aie.lock(%t73, 4) // a_pong
  %lock_73_b_ping = aie.lock(%t73, 5) // b_ping
  %lock_73_b_pong = aie.lock(%t73, 6) // b_pong

  %buf_74_aping = aie.buffer(%t74) {sym_name = "a74_ping" } : memref<64xi16>
  %buf_74_apong = aie.buffer(%t74) {sym_name = "a74_pong" } : memref<64xi16>
  %buf_74_bping = aie.buffer(%t74) {sym_name = "b74_ping" } : memref<64xi16>
  %buf_74_bpong = aie.buffer(%t74) {sym_name = "b74_pong" } : memref<64xi16>

  %lock_74_a_ping = aie.lock(%t74, 3) // a_ping
  %lock_74_a_pong = aie.lock(%t74, 4) // a_pong
  %lock_74_b_ping = aie.lock(%t74, 5) // b_ping
  %lock_74_b_pong = aie.lock(%t74, 6) // b_pong

  %buf_75_aping = aie.buffer(%t75) {sym_name = "a75_ping" } : memref<64xi16>
  %buf_75_apong = aie.buffer(%t75) {sym_name = "a75_pong" } : memref<64xi16>
  %buf_75_bping = aie.buffer(%t75) {sym_name = "b75_ping" } : memref<64xi16>
  %buf_75_bpong = aie.buffer(%t75) {sym_name = "b75_pong" } : memref<64xi16>

  %lock_75_a_ping = aie.lock(%t75, 3) // a_ping
  %lock_75_a_pong = aie.lock(%t75, 4) // a_pong
  %lock_75_b_ping = aie.lock(%t75, 5) // b_ping
  %lock_75_b_pong = aie.lock(%t75, 6) // b_pong

  aie.flow(%t70, DMA : 0, %t73, DMA : 0)
  aie.flow(%t73, DMA : 1, %t74, DMA : 0)
  aie.flow(%t74, DMA : 1, %t75, DMA : 0)
  aie.flow(%t75, DMA : 1, %t70, DMA : 0)

  func.func private @dequant_8x8(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func.func private @idct_8x8_mmult_h(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func.func private @idct_8x8_mmult_v(%A: memref<64xi16>, %B: memref<64xi16>) -> ()

  %c13 = aie.core(%t73) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    
    scf.for %iv = %lb to %ub step %step {
      aie.use_lock(%lock_73_a_ping, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_73_b_ping, "Acquire", 0) // acquire for write
      func.call @dequant_8x8(%buf_73_aping, %buf_73_bping) : (memref<64xi16>, memref<64xi16>) -> ()
      aie.use_lock(%lock_73_a_ping, "Release", 0) // release for write
      aie.use_lock(%lock_73_b_ping, "Release", 1) // release for read

      aie.use_lock(%lock_73_a_pong, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_73_b_pong, "Acquire", 0) // acquire for write
      func.call @dequant_8x8(%buf_73_apong, %buf_73_bpong) : (memref<64xi16>, memref<64xi16>) -> ()
      aie.use_lock(%lock_73_a_pong, "Release", 0) // release for write
      aie.use_lock(%lock_73_b_pong, "Release", 1) // release for read
    }

    aie.end
  } { link_with="dequant.o" }

  %c74 = aie.core(%t74) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    
    scf.for %iv = %lb to %ub step %step {
      aie.use_lock(%lock_74_a_ping, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_74_b_ping, "Acquire", 0) // acquire for write
      func.call @idct_8x8_mmult_h(%buf_74_aping, %buf_74_bping) : (memref<64xi16>, memref<64xi16>) -> ()
      aie.use_lock(%lock_74_a_ping, "Release", 0) // release for write
      aie.use_lock(%lock_74_b_ping, "Release", 1) // release for read

      aie.use_lock(%lock_74_a_pong, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_74_b_pong, "Acquire", 0) // acquire for write
      func.call @idct_8x8_mmult_h(%buf_74_apong, %buf_74_bpong) : (memref<64xi16>, memref<64xi16>) -> ()
      aie.use_lock(%lock_74_a_pong, "Release", 0) // release for write
      aie.use_lock(%lock_74_b_pong, "Release", 1) // release for read
    }

    aie.end
  } { link_with="idct_horizontal.o" }
  
    %c75 = aie.core(%t75) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    
    scf.for %iv = %lb to %ub step %step {
      aie.use_lock(%lock_75_a_ping, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_75_b_ping, "Acquire", 0) // acquire for write
      func.call @idct_8x8_mmult_v(%buf_75_aping, %buf_75_bping) : (memref<64xi16>, memref<64xi16>) -> ()
      aie.use_lock(%lock_75_a_ping, "Release", 0) // release for write
      aie.use_lock(%lock_75_b_ping, "Release", 1) // release for read

      aie.use_lock(%lock_75_a_pong, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_75_b_pong, "Acquire", 0) // acquire for write
      func.call @idct_8x8_mmult_v(%buf_75_apong, %buf_75_bpong) : (memref<64xi16>, memref<64xi16>) -> ()
      aie.use_lock(%lock_75_a_pong, "Release", 0) // release for write
      aie.use_lock(%lock_75_b_pong, "Release", 1) // release for read
    }

    aie.end
  } { link_with="idct_vertical.o" }

  // Tile DMA
  %m73 = aie.mem(%t73) {
      %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start("MM2S", 1, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%lock_73_a_ping, "Acquire", 0)
      aie.dma_bd(%buf_73_aping : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_73_a_ping, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%lock_73_a_pong, "Acquire", 0)
      aie.dma_bd(%buf_73_apong : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_73_a_pong, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%lock_73_b_ping, "Acquire", 1)
      aie.dma_bd(%buf_73_bping : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_73_b_ping, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%lock_73_b_pong, "Acquire", 1)
      aie.dma_bd(%buf_73_bpong : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_73_b_pong, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  // Tile DMA
  %m74 = aie.mem(%t74) {
      %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start("MM2S", 1, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%lock_74_a_ping, "Acquire", 0)
      aie.dma_bd(%buf_74_aping : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_74_a_ping, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%lock_74_a_pong, "Acquire", 0)
      aie.dma_bd(%buf_74_apong : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_74_a_pong, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%lock_74_b_ping, "Acquire", 1)
      aie.dma_bd(%buf_74_bping : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_74_b_ping, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%lock_74_b_pong, "Acquire", 1)
      aie.dma_bd(%buf_74_bpong : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_74_b_pong, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  // Tile DMA
  %m75 = aie.mem(%t75) {
      %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start("MM2S", 1, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%lock_75_a_ping, "Acquire", 0)
      aie.dma_bd(%buf_75_aping : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_75_a_ping, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%lock_75_a_pong, "Acquire", 0)
      aie.dma_bd(%buf_75_apong : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_75_a_pong, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%lock_75_b_ping, "Acquire", 1)
      aie.dma_bd(%buf_75_bping : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_75_b_ping, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%lock_75_b_pong, "Acquire", 1)
      aie.dma_bd(%buf_75_bpong : memref<64xi16>) { offset = 0 : i32, len = 64 : i32 }
      aie.use_lock(%lock_75_b_pong, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  // DDR buffer
  %buffer_in  = aie.external_buffer { sym_name = "buffer_in" } : memref<512 x i16>
  %buffer_out = aie.external_buffer { sym_name = "buffer_out" } : memref<512 x i16>

  %lock1 = aie.lock(%t70, 1) { sym_name = "buffer_in_lock" }
  %lock2 = aie.lock(%t70, 2) { sym_name = "buffer_out_lock" }

  // Shim DMA loads large buffer to local memory
  %dma = aie.shim_dma(%t70) {
      aie.dma_start(MM2S, 0, ^bd0, ^dma)
    ^dma:
      aie.dma_start(S2MM, 0, ^bd1, ^end)
    ^bd0:
      aie.use_lock(%lock1, "Acquire", 1)
      aie.dma_bd(%buffer_in : memref<512 x i16>) { offset = 0 : i32, len = 512 : i32 }
      aie.use_lock(%lock1, "Release", 0)
      aie.next_bd ^bd0
    ^bd1:
      aie.use_lock(%lock2, "Acquire", 1)
      aie.dma_bd(%buffer_out : memref<512 x i16>) { offset = 0 : i32, len = 512 : i32 }
      aie.use_lock(%lock2, "Release", 0)
      aie.next_bd ^bd1
    ^end:
      aie.end
  }
}
