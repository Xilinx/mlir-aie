//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: xchesscc -p me -P ${CARDANO}/data/cervino/lib -c %S/kernel.cc
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test_chess_04_deprecated_shim_dma_precompiled_kernel{

  %t74 = AIE.tile(7, 4)
  %t75 = AIE.tile(7, 5)

  %t73 = AIE.tile(7, 3)
  %t72 = AIE.tile(7, 2)
  %t71 = AIE.tile(7, 1)
  %t70 = AIE.tile(7, 0)

  %buf_73_aping = AIE.buffer(%t73) {sym_name = "a73_ping" } : memref<64xi16>
  %buf_73_apong = AIE.buffer(%t73) {sym_name = "a73_pong" } : memref<64xi16>
  %buf_73_bping = AIE.buffer(%t73) {sym_name = "b73_ping" } : memref<64xi16>
  %buf_73_bpong = AIE.buffer(%t73) {sym_name = "b73_pong" } : memref<64xi16>

  %lock_73_a_ping = AIE.lock(%t73, 3) // a_ping
  %lock_73_a_pong = AIE.lock(%t73, 4) // a_pong
  %lock_73_b_ping = AIE.lock(%t73, 5) // b_ping
  %lock_73_b_pong = AIE.lock(%t73, 6) // b_pong

  %buf_74_aping = AIE.buffer(%t74) {sym_name = "a74_ping" } : memref<64xi16>
  %buf_74_apong = AIE.buffer(%t74) {sym_name = "a74_pong" } : memref<64xi16>
  %buf_74_bping = AIE.buffer(%t74) {sym_name = "b74_ping" } : memref<64xi16>
  %buf_74_bpong = AIE.buffer(%t74) {sym_name = "b74_pong" } : memref<64xi16>

  %lock_74_a_ping = AIE.lock(%t74, 3) // a_ping
  %lock_74_a_pong = AIE.lock(%t74, 4) // a_pong
  %lock_74_b_ping = AIE.lock(%t74, 5) // b_ping
  %lock_74_b_pong = AIE.lock(%t74, 6) // b_pong

  %buf_75_aping = AIE.buffer(%t75) {sym_name = "a75_ping" } : memref<64xi16>
  %buf_75_apong = AIE.buffer(%t75) {sym_name = "a75_pong" } : memref<64xi16>
  %buf_75_bping = AIE.buffer(%t75) {sym_name = "b75_ping" } : memref<64xi16>
  %buf_75_bpong = AIE.buffer(%t75) {sym_name = "b75_pong" } : memref<64xi16>

  %lock_75_a_ping = AIE.lock(%t75, 3) // a_ping
  %lock_75_a_pong = AIE.lock(%t75, 4) // a_pong
  %lock_75_b_ping = AIE.lock(%t75, 5) // b_ping
  %lock_75_b_pong = AIE.lock(%t75, 6) // b_pong


  func private @dequant_8x8(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func private @idct_8x8_mmult_h(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func private @idct_8x8_mmult_v(%A: memref<64xi16>, %B: memref<64xi16>) -> ()


  func private @pass(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func private @func1(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func private @func2(%A: memref<64xi16>, %B: memref<64xi16>) -> ()
  func private @func3(%A: memref<64xi16>, %B: memref<64xi16>) -> ()

  // func private @idct_8x8_mmult_h(%A: memref<64xi16>, %B: memref<64xi16>) -> ()

  %c13 = AIE.core(%t73) { 
    %buffer_size =  arith.constant 64 : i32

    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    
    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    scf.for %iv = %lb to %ub step %step {
      
      AIE.useLock(%lock_73_a_ping, "Acquire", 1) // acquire for read
      AIE.useLock(%lock_73_b_ping, "Acquire", 0) // acquire for write
      call @dequant_8x8(%buf_73_aping, %buf_73_bping) : (memref<64xi16>, memref<64xi16>) -> ()
      AIE.useLock(%lock_73_a_ping, "Release", 0) // release for write
      AIE.useLock(%lock_73_b_ping, "Release", 1) // release for read

      AIE.useLock(%lock_73_a_pong, "Acquire", 1) // acquire for read
      AIE.useLock(%lock_73_b_pong, "Acquire", 0) // acquire for write
      call @dequant_8x8(%buf_73_apong, %buf_73_bpong) : (memref<64xi16>, memref<64xi16>) -> ()
      AIE.useLock(%lock_73_a_pong, "Release", 0) // release for write
      AIE.useLock(%lock_73_b_pong, "Release", 1) // release for read      
    }

    AIE.end

  } { link_with="dequant.o" }

  %c74 = AIE.core(%t74) { 
    %buffer_size =  arith.constant 64 : i32

    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    
    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    scf.for %iv = %lb to %ub step %step {
      
      AIE.useLock(%lock_74_a_ping, "Acquire", 1) // acquire for read
      AIE.useLock(%lock_74_b_ping, "Acquire", 0) // acquire for write
      call @pass(%buf_74_aping, %buf_74_bping) : (memref<64xi16>, memref<64xi16>) -> ()
      AIE.useLock(%lock_74_a_ping, "Release", 0) // release for write
      AIE.useLock(%lock_74_b_ping, "Release", 1) // release for read

      AIE.useLock(%lock_74_a_pong, "Acquire", 1) // acquire for read
      AIE.useLock(%lock_74_b_pong, "Acquire", 0) // acquire for write
      call @pass(%buf_74_apong, %buf_74_bpong) : (memref<64xi16>, memref<64xi16>) -> ()
      AIE.useLock(%lock_74_a_pong, "Release", 0) // release for write
      AIE.useLock(%lock_74_b_pong, "Release", 1) // release for read      
    }

    AIE.end
  } { link_with="pass.o" }


  
    %c75 = AIE.core(%t75) { 
    %buffer_size =  arith.constant 64 : i32

    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    
    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    scf.for %iv = %lb to %ub step %step {
      
      AIE.useLock(%lock_75_a_ping, "Acquire", 1) // acquire for read
      AIE.useLock(%lock_75_b_ping, "Acquire", 0) // acquire for write
      call @pass(%buf_75_aping, %buf_75_bping) : (memref<64xi16>, memref<64xi16>) -> ()
      AIE.useLock(%lock_75_a_ping, "Release", 0) // release for write
      AIE.useLock(%lock_75_b_ping, "Release", 1) // release for read

      AIE.useLock(%lock_75_a_pong, "Acquire", 1) // acquire for read
      AIE.useLock(%lock_75_b_pong, "Acquire", 0) // acquire for write
      call @pass(%buf_75_apong, %buf_75_bpong) : (memref<64xi16>, memref<64xi16>) -> ()
      AIE.useLock(%lock_75_a_pong, "Release", 0) // release for write
      AIE.useLock(%lock_75_b_pong, "Release", 1) // release for read      
    }

    AIE.end
  } { link_with="pass.o" }

  // Tile DMA
  %m73 = AIE.mem(%t73) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S1", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%lock_73_a_ping, "Acquire", 0)
      AIE.dmaBd(<%buf_73_aping : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_73_a_ping, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%lock_73_a_pong, "Acquire", 0)
      AIE.dmaBd(<%buf_73_apong : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_73_a_pong, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%lock_73_b_ping, "Acquire", 1)
      AIE.dmaBd(<%buf_73_bping : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_73_b_ping, "Release", 0)
      cf.br ^bd3
    ^bd3:
      AIE.useLock(%lock_73_b_pong, "Acquire", 1)
      AIE.dmaBd(<%buf_73_bpong : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_73_b_pong, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

    // Tile DMA
  %m74 = AIE.mem(%t74) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S1", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%lock_74_a_ping, "Acquire", 0)
      AIE.dmaBd(<%buf_74_aping : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_74_a_ping, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%lock_74_a_pong, "Acquire", 0)
      AIE.dmaBd(<%buf_74_apong : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_74_a_pong, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%lock_74_b_ping, "Acquire", 1)
      AIE.dmaBd(<%buf_74_bping : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_74_b_ping, "Release", 0)
      cf.br ^bd3
    ^bd3:
      AIE.useLock(%lock_74_b_pong, "Acquire", 1)
      AIE.dmaBd(<%buf_74_bpong : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_74_b_pong, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

  // Tile DMA
  %m75 = AIE.mem(%t75) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S1", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%lock_75_a_ping, "Acquire", 0)
      AIE.dmaBd(<%buf_75_aping : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_75_a_ping, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%lock_75_a_pong, "Acquire", 0)
      AIE.dmaBd(<%buf_75_apong : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_75_a_pong, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%lock_75_b_ping, "Acquire", 1)
      AIE.dmaBd(<%buf_75_bping : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_75_b_ping, "Release", 0)
      cf.br ^bd3
    ^bd3:
      AIE.useLock(%lock_75_b_pong, "Acquire", 1)
      AIE.dmaBd(<%buf_75_bpong : memref<64xi16>, 0, 64>, 0)
      AIE.useLock(%lock_75_b_pong, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

  // DDR buffer
  %buffer_in  = AIE.external_buffer 0x020100004000 : memref<512 x i16>
  %buffer_out = AIE.external_buffer 0x020100006000 : memref<512 x i16>

  // Shim DMA connection to kernel
  AIE.flow(%t71, "South" : 3, %t73, "DMA" : 0)

  %sw73  = AIE.switchbox(%t73) {
    AIE.connect<"DMA" : 1, "North" : 3>
  }
  %sw74  = AIE.switchbox(%t74) {
    AIE.connect<"South" : 3, "DMA" : 0>
    AIE.connect<"DMA" : 1, "North" : 3>
  }

  %sw75  = AIE.switchbox(%t75) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  AIE.flow(%t75, "DMA" : 1, %t71, "South" : 2)

  %sw1  = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 3>
    AIE.connect<"North" : 2, "South" : 2>
  }
  %mux1 = AIE.shimmux  (%t70) {
    AIE.connect<"DMA"   : 0, "North" : 3> 
    AIE.connect<"North" : 2, "DMA" : 0>
  }

  // Shim DMA loads large buffer to local memory
  %dma = AIE.shimDMA(%t70) {
      %lock1 = AIE.lock(%t70, 1)
      %lock2 = AIE.lock(%t70, 2)
      AIE.dmaStart(MM2S0, ^bd0, ^dma)
    ^dma:
      AIE.dmaStart(S2MM0, ^bd1, ^end)
    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_in : memref<512 x i16>, 0, 512>, 0)
      AIE.useLock(%lock1, Release, 0)
      cf.br ^bd0
    ^bd1:
      AIE.useLock(%lock2, Acquire, 1)
      AIE.dmaBd(<%buffer_out : memref<512 x i16>, 0, 512>, 0)
      AIE.useLock(%lock2, Release, 0)
      cf.br ^bd1
    ^end:
      AIE.end
  }


}
