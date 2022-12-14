//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf 


module @autocorrelation {
  %tile0_1 = AIE.tile(2, 1)
  %tile0_2 = AIE.tile(2, 2)
  %tile0_3 = AIE.tile(2, 3)
  %tile0_4 = AIE.tile(2, 4)
  %tile0_5 = AIE.tile(2, 5)
  %tile0_6 = AIE.tile(2, 6)
  %tile0_7 = AIE.tile(2, 7)
  %tile0_8 = AIE.tile(2, 8)

  %tile7_0 = AIE.tile(7, 0)

  %input = AIE.external_buffer {sym_name = "input"} : memref<1024 x i32>
  %output = AIE.external_buffer {sym_name = "output"} : memref<1024 x i32>

  %input_lock = AIE.lock(%tile7_0) {sym_name = "input_lock"}
  %output_lock = AIE.lock(%tile7_0) {sym_name = "output_lock"}

  %buf1_in = AIE.buffer(%tile0_1) : memref<1024xi32>
  %buf1_out = AIE.buffer(%tile0_1) : memref<1024xi32>
  %buf1_in_lock = AIE.lock(%tile0_1)
  %buf1_out_lock = AIE.lock(%tile0_1)

  %buf2_in = AIE.buffer(%tile0_2) : memref<1024xi32>
  %buf2_out = AIE.buffer(%tile0_2) : memref<1024xi32>
  %buf2_in_lock = AIE.lock(%tile0_2)
  %buf2_out_lock = AIE.lock(%tile0_2)

  %buf3_in = AIE.buffer(%tile0_3) : memref<1024xi32>
  %buf3_out = AIE.buffer(%tile0_3) : memref<1024xi32>
  %buf3_in_lock = AIE.lock(%tile0_3) 
  %buf3_out_lock = AIE.lock(%tile0_3) 

  %buf4_in = AIE.buffer(%tile0_4) : memref<1024xi32>
  %buf4_out = AIE.buffer(%tile0_4) : memref<1024xi32>
  %buf4_in_lock = AIE.lock(%tile0_4)
  %buf4_out_lock = AIE.lock(%tile0_4)

  AIE.shimDMA(%tile7_0) {
     AIE.dmaStart("MM2S", 0, ^bdin, ^dma2)
    ^dma2:
     AIE.dmaStart("S2MM", 0, ^bdout, ^end)
    ^bdin:
      AIE.useLock(%input_lock, "Acquire", 1)
      AIE.dmaBd(<%input : memref<1024xi32>, 0, 1024>, 0)
      AIE.useLock(%input_lock, "Release", 0)
      cf.br ^end
    ^bdout:
      AIE.useLock(%output_lock, "Acquire", 0)
      AIE.dmaBd(<%output : memref<1024xi32>, 0, 1024>, 0)
      AIE.useLock(%output_lock, "Release", 1)
      cf.br ^end
    ^end:
      AIE.end
  }

  // Simple broadcast flows for input
  AIE.flow(%tile7_0, DMA : 0, %tile0_1, DMA : 0)
  AIE.flow(%tile7_0, DMA : 0, %tile0_2, DMA : 0)
  AIE.flow(%tile7_0, DMA : 0, %tile0_3, DMA : 0)
  AIE.flow(%tile7_0, DMA : 0, %tile0_4, DMA : 0)

  // Flow for output
  AIE.flow(%tile0_1, DMA : 0, %tile7_0, DMA : 0)

  // The row 1 tile collects the result and DMAs back to the shim
  AIE.mem(%tile0_1)  {
    AIE.dmaStart("S2MM", 0, ^bd0, ^dma0)
  ^dma0: 
    AIE.dmaStart("MM2S", 0, ^bd1, ^end)
  ^bd0: 
    AIE.useLock(%buf1_in_lock, Acquire, 0)
    AIE.dmaBd(<%buf1_in : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%buf1_in_lock, Release, 1)
    cf.br ^end
  ^bd1: 
    AIE.useLock(%buf1_out_lock, Acquire, 1)
    AIE.dmaBd(<%buf1_out : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%buf1_out_lock, Release, 0)
    cf.br ^end
  ^end: 
    AIE.end
  }

  AIE.mem(%tile0_2)  {
    AIE.dmaStart("S2MM", 0, ^bd0, ^end)
  ^bd0: 
    AIE.useLock(%buf2_in_lock, Acquire, 0)
    AIE.dmaBd(<%buf2_in : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%buf2_in_lock, Release, 1)
    cf.br ^end
  ^end: 
    AIE.end
  }

  AIE.mem(%tile0_3)  {
    AIE.dmaStart("S2MM", 0, ^bd0, ^end)
  ^bd0: 
    AIE.useLock(%buf3_in_lock, Acquire, 0)
    AIE.dmaBd(<%buf3_in : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%buf3_in_lock, Release, 1)
    cf.br ^end
  ^end: 
    AIE.end
  }

  AIE.mem(%tile0_4)  {
    AIE.dmaStart("S2MM", 0, ^bd0, ^end)
  ^bd0: 
    AIE.useLock(%buf4_in_lock, Acquire, 0)
    AIE.dmaBd(<%buf4_in : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%buf4_in_lock, Release, 1)
    cf.br ^end
  ^end: 
    AIE.end
  }

  func.func @autocorrelate(%bufin: memref<1024xi32>, %bufout:memref<1024xi32>, %offset:index, %blocksize:index) -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cend = arith.constant 1024 : index
    %zero = arith.constant 0 : i32
  
    affine.for %arg1 = %c0 to %blocksize {
      %acc = affine.for %arg0 = %c0 to %cend
        iter_args(%acc_iter = %zero) -> (i32) {
        %a = affine.load %bufin[%arg0 + %arg1 + %offset] : memref<1024xi32>
        %b = affine.load %bufin[%arg0] : memref<1024xi32>
        %product = arith.muli %a, %b: i32
        %sum = arith.addi %acc_iter, %product: i32
        affine.yield %sum : i32
      }
      affine.store %acc, %bufout[%arg1] : memref<1024xi32>
    }

    return
  }

  AIE.core(%tile0_1) {
    AIE.useLock(%buf1_in_lock, "Acquire", 1)
    AIE.useLock(%buf1_out_lock, "Acquire", 0)
    %offset = arith.constant 0 : index
    %blocksize = arith.constant 16 : index
    func.call @autocorrelate(%buf1_in, %buf1_out, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()
    AIE.useLock(%buf1_in_lock, "Release", 0)

    // Append the prior results, block of 16.
    AIE.useLock(%buf2_out_lock, "Acquire", 1)
    %1 = memref.subview %buf2_out[0][64][1] : memref<1024xi32> to memref<64xi32>
    %2 = memref.subview %buf1_out[%blocksize][64][1] : memref<1024xi32> to memref<64xi32, strided<[1], offset: ?>>
    memref.copy %1, %2 : memref<64xi32> to memref<64xi32, strided<[1], offset: ?>>
    AIE.useLock(%buf2_out_lock, "Release", 0)

    AIE.useLock(%buf1_out_lock, "Release", 1)
    AIE.end
  }

  AIE.core(%tile0_2) {
    AIE.useLock(%buf2_in_lock, "Acquire", 1)
    AIE.useLock(%buf2_out_lock, "Acquire", 0)
    %offset = arith.constant 16 : index
    %blocksize = arith.constant 16 : index
    func.call @autocorrelate(%buf2_in, %buf2_out, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()
    AIE.useLock(%buf2_in_lock, "Release", 0)

    // Append the prior results, block of 16.
    AIE.useLock(%buf3_out_lock, "Acquire", 1)
    %1 = memref.subview %buf3_out[0][64][1] : memref<1024xi32> to memref<64xi32>
    %2 = memref.subview %buf2_out[%blocksize][64][1] : memref<1024xi32> to memref<64xi32, strided<[1], offset: ?>>
    memref.copy %1, %2 : memref<64xi32> to memref<64xi32, strided<[1], offset: ?>>
    AIE.useLock(%buf3_out_lock, "Release", 0)

    AIE.useLock(%buf2_out_lock, "Release", 1)
    AIE.end
  }

  AIE.core(%tile0_3) {
    AIE.useLock(%buf3_in_lock, "Acquire", 1)
    AIE.useLock(%buf3_out_lock, "Acquire", 0)
    %offset = arith.constant 32 : index
    %blocksize = arith.constant 16 : index
    func.call @autocorrelate(%buf3_in, %buf3_out, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()
    AIE.useLock(%buf3_in_lock, "Release", 0)

    // Append the prior results, block of 16.
    AIE.useLock(%buf4_out_lock, "Acquire", 1)
    %1 = memref.subview %buf4_out[0][64][1] : memref<1024xi32> to memref<64xi32>
    %2 = memref.subview %buf3_out[%blocksize][64][1] : memref<1024xi32> to memref<64xi32, strided<[1], offset: ?>>
    memref.copy %1, %2 : memref<64xi32> to memref<64xi32, strided<[1], offset: ?>>
    AIE.useLock(%buf4_out_lock, "Release", 0)

    AIE.useLock(%buf3_out_lock, "Release", 1)
    AIE.end
  }

  AIE.core(%tile0_4) {
    AIE.useLock(%buf4_in_lock, "Acquire", 1)
    AIE.useLock(%buf4_out_lock, "Acquire", 0)
    %offset = arith.constant 48 : index
    %blocksize = arith.constant 16 : index
    func.call @autocorrelate(%buf4_in, %buf4_out, %offset, %blocksize) : (memref<1024xi32>, memref<1024xi32>, index, index) -> ()
    AIE.useLock(%buf4_in_lock, "Release", 0)
    AIE.useLock(%buf4_out_lock, "Release", 1)
    AIE.end
  }

}
