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
    // CHECK: %{{.*}}tile_0_0 = aie.tile(0, 0)
    // CHECK: %{{.*}}tile_0_1 = aie.tile(0, 1)
    // CHECK: %{{.*}}tile_0_2 = aie.tile(0, 2)
    // CHECK: %[[VAL_2:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "objFifo_out1_cons_buff_0"} : memref<64x64xi8>
    // CHECK: %[[VAL_3:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "objFifo_out1_cons_buff_1"} : memref<64x64xi8>
    // CHECK: %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_0_1, 2) {init = 2 : i32, sym_name = "objFifo_out1_cons_prod_lock_0"}
    // CHECK: %[[VAL_5:.*]] = aie.lock(%{{.*}}tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_cons_lock_0"}
    // CHECK: %[[VAL_6:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8>
    // CHECK: %[[VAL_7:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8>
    // CHECK: %[[VAL_8:.*]] = aie.lock(%{{.*}}tile_0_2, 2) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock_0"}
    // CHECK: %[[VAL_9:.*]] = aie.lock(%{{.*}}tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock_0"}
    // CHECK: %[[VAL_10:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8>
    // CHECK: %[[VAL_11:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8>
    // CHECK: %[[VAL_12:.*]] = aie.lock(%{{.*}}tile_0_2, 0) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock_0"}
    // CHECK: %[[VAL_13:.*]] = aie.lock(%{{.*}}tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock_0"}
    // CHECK: %[[VAL_14:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "objFifo_in1_buff_0"} : memref<64x64xi8>
    // CHECK: %[[VAL_15:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "objFifo_in1_buff_1"} : memref<64x64xi8>
    // CHECK: %[[VAL_16:.*]] = aie.lock(%{{.*}}tile_0_1, 0) {init = 2 : i32, sym_name = "objFifo_in1_prod_lock_0"}
    // CHECK: %[[VAL_17:.*]] = aie.lock(%{{.*}}tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_lock_0"}
    // CHECK: aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_1, DMA : 0)
    // CHECK: aie.flow(%{{.*}}tile_0_1, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
    // CHECK: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_1, DMA : 1)
    // CHECK: aie.flow(%{{.*}}tile_0_1, DMA : 1, %{{.*}}tile_0_0, DMA : 0)
    // CHECK: %core_0_2 = aie.core(%{{.*}}tile_0_2) {
    // CHECK:   aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, 1)
    // CHECK:   aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
    // CHECK:   %c0 = arith.constant 0 : index
    // CHECK:   %c1 = arith.constant 1 : index
    // CHECK:   %c64 = arith.constant 64 : index
    // CHECK:   %c12_i8 = arith.constant 12 : i8
    // CHECK:   scf.for %arg0 = %c0 to %c64 step %c1 {
    // CHECK:     scf.for %arg1 = %c0 to %c64 step %c1 {
    // CHECK:       %0 = memref.load %[[VAL_10]][%arg0, %arg1] : memref<64x64xi8>
    // CHECK:       %1 = arith.addi %0, %c12_i8 : i8
    // CHECK:       memref.store %1, %[[VAL_10]][%arg0, %arg1] : memref<64x64xi8>
    // CHECK:     }
    // CHECK:   }
    // CHECK:   aie.use_lock(%[[VAL_12]], Release, 1)
    // CHECK:   aie.use_lock(%[[VAL_9]], Release, 1)
    // CHECK:   aie.end
    // CHECK: }
    // CHECK: aiex.runtime_sequence(%arg0: memref<61x56xi8>, %arg1: memref<32xi8>, %arg2: memref<64x64xi8>) {
    // CHECK:   aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 61, 56][0, 0, 56, 1]) {id = 0 : i64, metadata = @objFifo_in0} : memref<61x56xi8>
    // CHECK:   aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64, 1]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0} : memref<64x64xi8>
    // CHECK:   aiex.npu.dma_wait {symbol = @objFifo_out0}
    // CHECK: }
    // CHECK: aie.shim_dma_allocation @objFifo_in0(MM2S, 0, 0)
    // CHECK: %memtile_dma_0_1 = aie.memtile_dma(%{{.*}}tile_0_1) {
    // CHECK:   %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    // CHECK: ^bb1:
    // CHECK:   aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_14]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_17]], Release, 1)
    // CHECK:   aie.next_bd ^bb2
    // CHECK: ^bb2:
    // CHECK:   aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_15]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_17]], Release, 1)
    // CHECK:   aie.next_bd ^bb1
    // CHECK: ^bb3:
    // CHECK:   %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    // CHECK: ^bb4:
    // CHECK:   aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_14]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_16]], Release, 1)
    // CHECK:   aie.next_bd ^bb5
    // CHECK: ^bb5:
    // CHECK:   aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_15]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_16]], Release, 1)
    // CHECK:   aie.next_bd ^bb4
    // CHECK: ^bb6:
    // CHECK:   %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    // CHECK: ^bb7:
    // CHECK:   aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_2]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_5]], Release, 1)
    // CHECK:   aie.next_bd ^bb8
    // CHECK: ^bb8:
    // CHECK:   aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_3]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_5]], Release, 1)
    // CHECK:   aie.next_bd ^bb7
    // CHECK: ^bb9:
    // CHECK:   %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    // CHECK: ^bb10:
    // CHECK:   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_2]] : memref<64x64xi8>, 0, 4096, [<size = 61, stride = 56>, <size = 56, stride = 1>], [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])
    // CHECK:   aie.use_lock(%[[VAL_4]], Release, 1)
    // CHECK:   aie.next_bd ^bb11
    // CHECK: ^bb11:
    // CHECK:   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_3]] : memref<64x64xi8>, 0, 4096, [<size = 61, stride = 56>, <size = 56, stride = 1>], [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])
    // CHECK:   aie.use_lock(%[[VAL_4]], Release, 1)
    // CHECK:   aie.next_bd ^bb10
    // CHECK: ^bb12:
    // CHECK:   aie.end
    // CHECK: }
    // CHECK: %mem_0_2 = aie.mem(%{{.*}}tile_0_2) {
    // CHECK:   %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    // CHECK: ^bb1:
    // CHECK:   aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_10]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_13]], Release, 1)
    // CHECK:   aie.next_bd ^bb2
    // CHECK: ^bb2:
    // CHECK:   aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_11]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_13]], Release, 1)
    // CHECK:   aie.next_bd ^bb1
    // CHECK: ^bb3:
    // CHECK:   %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    // CHECK: ^bb4:
    // CHECK:   aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_6]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_8]], Release, 1)
    // CHECK:   aie.next_bd ^bb5
    // CHECK: ^bb5:
    // CHECK:   aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
    // CHECK:   aie.dma_bd(%[[VAL_7]] : memref<64x64xi8>, 0, 4096)
    // CHECK:   aie.use_lock(%[[VAL_8]], Release, 1)
    // CHECK:   aie.next_bd ^bb4
    // CHECK: ^bb6:
    // CHECK:   aie.end
    // CHECK:   }
    // CHECK: aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)

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
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 61, 56][0, 0, 56, 1]) {id = 0 : i64, metadata = @objFifo_in0} : memref<61x56xi8>
      aiex.npu.dma_memcpy_nd (%arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64, 1]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<64x64xi8>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }
}
