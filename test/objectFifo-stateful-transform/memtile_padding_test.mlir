//===- memtile_padding_test.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s
    // CHECK: %tile_0_0 = aie.tile(0, 0)
    // CHECK: %tile_0_1 = aie.tile(0, 1)
    // CHECK: %tile_0_2 = aie.tile(0, 2)
    // CHECK: %objFifo_out0_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 1 : i32, sym_name = "objFifo_out0_cons_prod_lock"}
    // CHECK: %objFifo_out0_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_cons_lock"}
    // CHECK: %objFifo_out1_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out1_cons_buff_0"} : memref<64x64xi8> 
    // CHECK: %objFifo_out1_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out1_cons_buff_1"} : memref<64x64xi8> 
    // CHECK: %objFifo_out1_cons_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "objFifo_out1_cons_prod_lock"}
    // CHECK: %objFifo_out1_cons_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_cons_lock"}
    // CHECK: %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8> 
    // CHECK: %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8> 
    // CHECK: %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock"}
    // CHECK: %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}
    // CHECK: %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8> 
    // CHECK: %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8> 
    // CHECK: %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
    // CHECK: %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
    // CHECK: %objFifo_in1_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in1_buff_0"} : memref<64x64xi8> 
    // CHECK: %objFifo_in1_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in1_buff_1"} : memref<64x64xi8> 
    // CHECK: %objFifo_in1_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32, sym_name = "objFifo_in1_prod_lock"}
    // CHECK: %objFifo_in1_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_lock"}
    // CHECK: %objFifo_in0_prod_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "objFifo_in0_prod_lock"}
    // CHECK: %objFifo_in0_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_lock"}
    // CHECK: aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    // CHECK: aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    // CHECK: aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    // CHECK: aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    // CHECK: %core_0_2 = aie.core(%tile_0_2) {
    // CHECK:   aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
    // CHECK:   %c0 = arith.constant 0 : index
    // CHECK:   %c1 = arith.constant 1 : index
    // CHECK:   %c64 = arith.constant 64 : index
    // CHECK:   %c12_i8 = arith.constant 12 : i8
    // CHECK:   scf.for %arg0 = %c0 to %c64 step %c1 {
    // CHECK:     scf.for %arg1 = %c0 to %c64 step %c1 {
    // CHECK:       %0 = memref.load %objFifo_in1_cons_buff_0[%arg0, %arg1] : memref<64x64xi8>
    // CHECK:       %1 = arith.addi %0, %c12_i8 : i8
    // CHECK:       memref.store %1, %objFifo_in1_cons_buff_0[%arg0, %arg1] : memref<64x64xi8>
    // CHECK:     }
    // CHECK:   }
    // CHECK:   aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
    // CHECK:   aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
    // CHECK:   aie.end
    // CHECK: }
    // CHECK: aie.shim_dma_allocation @objFifo_in0(MM2S, 0, 0)
    // CHECK: aiex.runtime_sequence(%arg0: memref<61x56xi8>, %arg1: memref<32xi8>, %arg2: memref<64x64xi8>) {
    // CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 61, 56][0, 0, 56, 1]) {id = 0 : i64, metadata = @objFifo_in0} : memref<61x56xi8>
    // CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64, 1]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0} : memref<64x64xi8>
    // CHECK:   aiex.npu.dma_wait {symbol = @objFifo_out0}
    // CHECK: }
    // CHECK: %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
    // CHECK:   %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    // CHECK: ^bb1: 
    // CHECK:   aie.use_lock(%objFifo_in1_prod_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_in1_buff_0 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_in1_cons_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb2
    // CHECK: ^bb2: 
    // CHECK:   aie.use_lock(%objFifo_in1_prod_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_in1_buff_1 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_in1_cons_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb1
    // CHECK: ^bb3:
    // CHECK:   %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    // CHECK: ^bb4:
    // CHECK:   aie.use_lock(%objFifo_in1_cons_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_in1_buff_0 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_in1_prod_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb5
    // CHECK: ^bb5: 
    // CHECK:   aie.use_lock(%objFifo_in1_cons_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_in1_buff_1 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_in1_prod_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb4
    // CHECK: ^bb6: 
    // CHECK:   %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    // CHECK: ^bb7: 
    // CHECK:   aie.use_lock(%objFifo_out1_cons_prod_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_out1_cons_buff_0 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_out1_cons_cons_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb8
    // CHECK: ^bb8:
    // CHECK:   aie.use_lock(%objFifo_out1_cons_prod_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_out1_cons_buff_1 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_out1_cons_cons_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb7
    // CHECK: ^bb9:
    // CHECK:   %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    // CHECK: ^bb10: 
    // CHECK:   aie.use_lock(%objFifo_out1_cons_cons_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_out1_cons_buff_0 : memref<64x64xi8>, 0, 4096, [<size = 61, stride = 56>, <size = 56, stride = 1>], [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])
    // CHECK:   aie.use_lock(%objFifo_out1_cons_prod_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb11
    // CHECK: ^bb11:
    // CHECK:   aie.use_lock(%objFifo_out1_cons_cons_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_out1_cons_buff_1 : memref<64x64xi8>, 0, 4096, [<size = 61, stride = 56>, <size = 56, stride = 1>], [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])
    // CHECK:   aie.use_lock(%objFifo_out1_cons_prod_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb10
    // CHECK: ^bb12:
    // CHECK:   aie.end
    // CHECK: }
    // CHECK: aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)
    // CHECK: %mem_0_2 = aie.mem(%tile_0_2) {
    // CHECK:   %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    // CHECK: ^bb1: 
    // CHECK:   aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb2
    // CHECK: ^bb2:
    // CHECK:   aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_in1_cons_buff_1 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb1
    // CHECK: ^bb3: 
    // CHECK:   %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    // CHECK: ^bb4: 
    // CHECK:   aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_out1_buff_0 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb5
    // CHECK: ^bb5: 
    // CHECK:   aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%objFifo_out1_buff_1 : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
    // CHECK:   aie.next_bd ^bb4
    // CHECK: ^bb6:
    // CHECK:   aie.end
    // CHECK:   }

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @objFifo_in0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<56x56xi8>>
    aie.objectfifo @objFifo_in1(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])
    aie.objectfifo @objFifo_out1(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>>
    aie.objectfifo @objFifo_out0(%tile_0_1 dimensionsToStream [<size = 61, stride = 56>, <size = 56, stride = 1>], {%tile_0_0}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>]>} : !aie.objectfifo<memref<64x64xi8>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %subview = aie.objectfifo.acquire @objFifo_in1 (Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
      %subview1 = aie.objectfifo.acquire @objFifo_out1 (Produce, 1) : !aie.objectfifosubview<memref<64x64xi8>>
      %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
      %elem1 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
      
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c12_i8 = arith.constant 12 : i8
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %elem[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c12_i8 : i8
          memref.store %1, %elem1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.objectfifo.release @objFifo_in1 (Consume, 1)
      aie.objectfifo.release @objFifo_out1 (Produce, 1)
      aie.end
    }

    aiex.runtime_sequence(%arg0: memref<61x56xi8>, %arg1: memref<32xi8>, %arg2: memref<64x64xi8>) {
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 61, 56][0, 0, 56, 1]) {id = 0 : i64, metadata = @objFifo_in0} : memref<61x56xi8>
      aiex.npu.dma_memcpy_nd (0, 0, %arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64, 1]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<64x64xi8>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }
}