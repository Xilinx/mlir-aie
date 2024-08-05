//===- shims.mlir ----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true | FileCheck %s

// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000001D000  Size: 8
// CHECK:     Address: 0x000000000001D000  Data@ {{0x[0-9a-z]+}} is: 0x00000004 
// CHECK:     Address: 0x000000000001D004  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D008  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D00C  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D010  Data@ {{0x[0-9a-z]+}} is: 0x80000000 
// CHECK:     Address: 0x000000000001D014  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D018  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000001D01C  Data@ {{0x[0-9a-z]+}} is: 0x02000000 

// CHECK: (Write64): Address:  0x000000000001D204 Data:  0x80000000  
// CHECK: (MaskWrite64): Address: 0x000000000001D200  Mask: 0x00000000  Data: 0x00000001 
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000201D000  Size: 8
// CHECK:     Address: 0x000000000201D000  Data@ {{0x[0-9a-z]+}} is: 0x00000004 
// CHECK:     Address: 0x000000000201D004  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000201D008  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000201D00C  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000201D010  Data@ {{0x[0-9a-z]+}} is: 0x80000000 
// CHECK:     Address: 0x000000000201D014  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000201D018  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000201D01C  Data@ {{0x[0-9a-z]+}} is: 0x02000000 

// CHECK: (Write64): Address:  0x000000000201D204 Data:  0x80000000  
// CHECK: (MaskWrite64): Address: 0x000000000201D200  Mask: 0x00000000  Data: 0x00000001 
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000401D000  Size: 8
// CHECK:     Address: 0x000000000401D000  Data@ {{0x[0-9a-z]+}} is: 0x00000010 
// CHECK:     Address: 0x000000000401D004  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D008  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D00C  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D010  Data@ {{0x[0-9a-z]+}} is: 0x80000000 
// CHECK:     Address: 0x000000000401D014  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D018  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D01C  Data@ {{0x[0-9a-z]+}} is: 0x02041000 

// CHECK: (NOP Command): Payload Length: 0 
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000401D020  Size: 8
// CHECK:     Address: 0x000000000401D020  Data@ {{0x[0-9a-z]+}} is: 0x00000004 
// CHECK:     Address: 0x000000000401D024  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D028  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D02C  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D030  Data@ {{0x[0-9a-z]+}} is: 0x80000000 
// CHECK:     Address: 0x000000000401D034  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D038  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000401D03C  Data@ {{0x[0-9a-z]+}} is: 0x02000000 

// CHECK: (Write64): Address:  0x000000000401D204 Data:  0x80000000  
// CHECK: (MaskWrite64): Address: 0x000000000401D200  Mask: 0x00000000  Data: 0x00000001 
// CHECK: (Write64): Address:  0x000000000401D214 Data:  0x00000001  
// CHECK: (MaskWrite64): Address: 0x000000000401D210  Mask: 0x00000000  Data: 0x00000001 
// CHECK: (NOP Command): Payload Length: 2 
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000601D000  Size: 8
// CHECK:     Address: 0x000000000601D000  Data@ {{0x[0-9a-z]+}} is: 0x00000004 
// CHECK:     Address: 0x000000000601D004  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000601D008  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000601D00C  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000601D010  Data@ {{0x[0-9a-z]+}} is: 0x80000000 
// CHECK:     Address: 0x000000000601D014  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000601D018  Data@ {{0x[0-9a-z]+}} is: 0x00000000 
// CHECK:     Address: 0x000000000601D01C  Data@ {{0x[0-9a-z]+}} is: 0x02000000 

// CHECK: (Write64): Address:  0x000000000601D204 Data:  0x80000000  
// CHECK: (MaskWrite64): Address: 0x000000000601D200  Mask: 0x00000000  Data: 0x00000001 
// CHECK: (Write64): Address:  0x000000000403F008 Data:  0x80000000  
// CHECK: (Write64): Address:  0x000000000403F100 Data:  0x80000000  
// CHECK: (Write64): Address:  0x000000000403F010 Data:  0x8000000E  
// CHECK: (Write64): Address:  0x000000000403F138 Data:  0x80000000  
// CHECK: (MaskWrite64): Address: 0x000000000001F004  Mask: 0x00000030  Data: 0x00000010 
// CHECK: (MaskWrite64): Address: 0x000000000201F004  Mask: 0x00000030  Data: 0x00000010 
// CHECK: (MaskWrite64): Address: 0x000000000401F004  Mask: 0x00000030  Data: 0x00000010 
// CHECK: (MaskWrite64): Address: 0x000000000601F004  Mask: 0x00000030  Data: 0x00000010 

module {
 aie.device(npu1_4col) {
  %buffer = aie.external_buffer { sym_name = "buf" } : memref<16 x f32>
  %t00 = aie.tile(0, 0)
  %t10 = aie.tile(1, 0)
  %t20 = aie.tile(2, 0)
  %t30 = aie.tile(3, 0)
  aie.shim_mux(%t00)  {
    aie.connect<North : 2, DMA : 0>
  }
  aie.shim_dma(%t00)  {
      aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buffer : memref<16 x f32>) {len = 4 : i32, bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
  }
  aie.shim_mux(%t10)  {
    aie.connect<North : 2, DMA : 0>
  }
  aie.shim_dma(%t10)  {
      aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buffer : memref<16 x f32>) {len = 4 : i32, bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
  }
  %s20 = aie.switchbox(%t20)  {
    aie.connect<North : 0, South : 2>
  }
  %mux = aie.shim_mux(%t20)  {
    aie.connect<North : 2, DMA : 0>
  }
  %dma = aie.shim_dma(%t20)  {
      %lock0 = aie.lock(%t20, 0)
      %lock1 = aie.lock(%t20, 1)

      aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      aie.dma_start(MM2S, 0, ^bd1, ^end)
    ^bd0:
      aie.use_lock(%lock0, Acquire, 0)
      aie.dma_bd(%buffer : memref<16 x f32>) {len = 16 : i32, bd_id = 0 : i32}
      aie.use_lock(%lock0, Release, 1)
      aie.next_bd ^bd0
    ^bd1:
      // aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer : memref<16 x f32>) {len = 4 : i32, bd_id = 1 : i32}
      // aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd1
    ^end:
      aie.end
  }
  aie.shim_mux(%t30)  {
    aie.connect<North : 2, DMA : 0>
  }
  aie.shim_dma(%t30)  {
      aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buffer : memref<16 x f32>) {len = 4 : i32, bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
  }
}
}