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
// xchesscc -p me -P /proj/xbuilds/2021.1_released/installs/lin64/Vitis/2021.1/aietools/data/cervino/lib -L/proj/xbuilds/2021.1_released/installs/lin64/Vitis/2021.1/cardano/lib -c ./hdiff_lap.cc ./hdiff_flux.cc

//  aiecc.py --sysroot=/group/xrlabs/platforms/pynq_on_versal_vck190_hacked/vck190-sysroot aie.mlir -v -I/scratch/gagandee/acdc-aie/runtime_lib/ /scratch/gagandee/acdc-aie/runtime_lib/test_library.cpp ./test.cpp -v -o test.elf
   

module @test_chess_04_deprecated_shim_dma_precompiled_kernel{


  %t74 = AIE.tile(7, 4)//flux
  %t73 = AIE.tile(7, 3) //lap

  // %t72 = AIE.tile(7, 2)//flux
  // %t71 = AIE.tile(7, 1) //lap
  %t70 = AIE.tile(7, 0) //shimDMA

  %buf_73a_ping = AIE.buffer(%t73) {sym_name = "a73_ping" } : memref<640xi32> //in
  %buf_73a_pong = AIE.buffer(%t73) {sym_name = "a73_pong" } : memref<640xi32> //in
  %buf_73b_ping = AIE.buffer(%t73) {sym_name = "b73_ping" } : memref<640xi32> //out
  %buf_73b_pong = AIE.buffer(%t73) {sym_name = "b73_pong" } : memref<640xi32> //out

  %lock_73a_ping = AIE.lock(%t73, 3) // a_ping
  %lock_73a_pong = AIE.lock(%t73, 4) // a_pong
  %lock_73b_ping = AIE.lock(%t73, 5) // b_ping
  %lock_73b_pong = AIE.lock(%t73, 6) // b_pong

  %buf_74a_ping = AIE.buffer(%t74) {sym_name = "a74_ping" } : memref<640xi32>
  %buf_74a_pong = AIE.buffer(%t74) {sym_name = "a74_pong" } : memref<640xi32>
  %buf_74b_ping = AIE.buffer(%t74) {sym_name = "b74_ping" } : memref<640xi32>
  %buf_74b_pong = AIE.buffer(%t74) {sym_name = "b74_pong" } : memref<640xi32>

  %buf_74c_ping = AIE.buffer(%t74) {sym_name = "c74_ping" } : memref<640xi32>
  %buf_74c_pong = AIE.buffer(%t74) {sym_name = "c74_pong" } : memref<640xi32>

  %lock_74a_ping = AIE.lock(%t74, 3) // a_ping
  %lock_74a_pong = AIE.lock(%t74, 4) // a_pong
  %lock_74b_ping = AIE.lock(%t74, 5) // b_ping
  %lock_74b_pong = AIE.lock(%t74, 6) // b_pong

  %lock_74c_ping = AIE.lock(%t74, 0) // c_ping
  %lock_74c_pong = AIE.lock(%t74, 1) // c_pong

  func private @hdiff_lap(%A: memref<640xi32>, %C: memref<640xi32>) -> ()
  func private @hdiff_flux(%A: memref<640xi32>, %C: memref<640xi32>, %B: memref<640xi32>) -> ()

  %c13 = AIE.core(%t73) { 
    
    %lb = constant 0 : index
    %ub = constant 1 : index
    %step = constant 1 : index
    
    scf.for %iv = %lb to %ub step %step {
      
      AIE.useLock(%lock_73a_ping, "Acquire", 1, 0) // acquire for read
      AIE.useLock(%lock_73b_ping, "Acquire", 0, 0) // acquire for write
      call @hdiff_lap(%buf_73a_ping, %buf_73b_ping) : (memref<640xi32>, memref<640xi32>) -> ()
      AIE.useLock(%lock_73a_ping, "Release", 0, 0) // release for write
      AIE.useLock(%lock_73b_ping, "Release", 1, 0) // release for read

      AIE.useLock(%lock_73a_pong, "Acquire", 1, 0) // acquire for read
      AIE.useLock(%lock_73b_pong, "Acquire", 0, 0) // acquire for write
      call @hdiff_lap(%buf_73a_pong, %buf_73b_pong) : (memref<640xi32>, memref<640xi32>) -> ()
      AIE.useLock(%lock_73a_pong, "Release", 0, 0) // release for write
      AIE.useLock(%lock_73b_pong, "Release", 1, 0) // release for read
      
    }

    AIE.end
  } { link_with="hdiff_lap.o" }


  %c74 = AIE.core(%t74) { 
    // %buffer_size =  arith.constant 64 : i32

    %lb = constant 0 : index
    %ub = constant 1 : index
    %step = constant 1 : index

    scf.for %iv = %lb to %ub step %step {
      
      AIE.useLock(%lock_74a_ping, "Acquire", 1, 0) // acquire for read
      AIE.useLock(%lock_74c_ping, "Acquire", 1, 0) // acquire for read
      AIE.useLock(%lock_74b_ping, "Acquire", 0, 0) // acquire for write
      call @hdiff_flux(%buf_74a_pong,%buf_74c_pong, %buf_74b_pong) : (memref<640xi32>,memref<640xi32>, memref<640xi32>) -> ()
      AIE.useLock(%lock_74a_ping, "Release", 0, 0) // release for write
      AIE.useLock(%lock_74c_ping, "Release", 0, 0) // release for write
      AIE.useLock(%lock_74b_ping, "Release", 1, 0) // release for read

      AIE.useLock(%lock_74a_pong, "Acquire", 1, 0) // acquire for read
       AIE.useLock(%lock_74c_pong, "Acquire", 1, 0) // acquire for read
      AIE.useLock(%lock_74b_pong, "Acquire", 0, 0) // acquire for write
      call @hdiff_flux(%buf_74a_pong, %buf_74c_pong, %buf_74b_pong) : (memref<640xi32>,memref<640xi32>, memref<640xi32>) -> ()
      AIE.useLock(%lock_74a_pong, "Release", 0, 0) // release for write
      AIE.useLock(%lock_74c_pong, "Release", 0, 0) // release for write
      AIE.useLock(%lock_74b_pong, "Release", 1, 0) // release for read      
    }

    AIE.end
  } { link_with="hdiff_flux.o" }

  // Tile DMA
  %m73 = AIE.mem(%t73) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%lock_73a_ping, "Acquire", 0, 0)
      AIE.dmaBd(<%buf_73a_ping : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_73a_ping, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%lock_73a_pong, "Acquire", 0, 0)
      AIE.dmaBd(<%buf_73a_pong : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_73a_pong, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%lock_73b_ping, "Acquire", 1, 0)
      AIE.dmaBd(<%buf_73b_ping : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_73b_ping, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%lock_73b_pong, "Acquire", 1, 0)
      AIE.dmaBd(<%buf_73b_pong : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_73b_pong, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }



  %m74 = AIE.mem(%t74) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma1)
    ^dma1:
      %srcDma2 = AIE.dmaStart("S2MM1", ^bd00, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%lock_74a_ping, "Acquire", 0, 0)
      AIE.dmaBd(<%buf_74a_ping : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_74a_ping, "Release", 1, 0)
       br ^bd1
    ^bd1:
      AIE.useLock(%lock_74a_pong, "Acquire", 0, 0)
      AIE.dmaBd(<%buf_74a_pong : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_74a_pong, "Release", 1, 0)
     br ^bd0
    ^bd00:
      AIE.useLock(%lock_74c_ping, "Acquire", 0, 0)
      AIE.dmaBd(<%buf_74c_ping : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_74c_ping, "Release", 1, 0)
      br ^bd11
    ^bd11:
      AIE.useLock(%lock_74c_pong, "Acquire", 0, 0)
      AIE.dmaBd(<%buf_74c_pong : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_74c_pong, "Release", 1, 0)
      br ^bd00

    ^bd2:
      AIE.useLock(%lock_74b_ping, "Acquire", 1, 0)
      AIE.dmaBd(<%buf_74b_ping : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_74b_ping, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%lock_74b_pong, "Acquire", 1, 0)
      AIE.dmaBd(<%buf_74b_pong : memref<640xi32>, 0, 640>, 0)
      AIE.useLock(%lock_74b_pong, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  // DDR buffer
  %buffer_in  = AIE.external_buffer 0x020100004000 : memref<640 x i32>
  %buffer_out = AIE.external_buffer 0x020100006000 : memref<640 x i32>

  // Shim DMA connection to kernel
  AIE.flow(%t70, "DMA" : 0, %t73, "DMA" : 0)
  AIE.flow(%t70, "DMA" : 1, %t74, "DMA" : 0)
  AIE.flow(%t73, "DMA" : 0, %t74, "DMA" : 1)
  AIE.flow(%t74, "DMA" : 0, %t70, "DMA" : 0)


%lock1 = AIE.lock(%t70, 1)
%lock11 = AIE.lock(%t70, 3)
//      AIE.dmaStart(MM2S0, ^bd0, ^end)
%lock2 = AIE.lock(%t70, 2)
  // Shim DMA loads large buffer to local memory
  
  %dma = AIE.shimDMA(%t70) {
      AIE.dmaStart(MM2S0, ^bd0, ^dma1)
    ^dma1:
      AIE.dmaStart(MM2S1, ^bd00, ^dma)
    ^dma:
      AIE.dmaStart(S2MM0, ^bd1, ^end)
    ^bd0:
      AIE.useLock(%lock1, Acquire, 1, 0)
      AIE.dmaBd(<%buffer_in : memref<640 x i32>, 0, 640>, 0)
      AIE.useLock(%lock1, Release, 0, 0)
      br ^bd0
    ^bd00:
      AIE.useLock(%lock11, Acquire, 1, 0)
      AIE.dmaBd(<%buffer_in : memref<640 x i32>, 0, 640>, 0)
      AIE.useLock(%lock11, Release, 0, 0)
//      br ^bd0
      br ^bd00
    ^bd1:
      AIE.useLock(%lock2, Acquire, 1, 0)
      AIE.dmaBd(<%buffer_out : memref<640 x i32>, 0, 640>, 0)
      AIE.useLock(%lock2, Release, 0, 0)
     br ^bd1
      // br ^end
    ^end:
      AIE.end
  }


}