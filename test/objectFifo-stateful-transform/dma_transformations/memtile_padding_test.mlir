//===- memtile_padding_test.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_2]]) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objFifo_out1_cons_buff_0"} : memref<64x64xi8>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objFifo_out1_cons_buff_1"} : memref<64x64xi8>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "objFifo_out1_cons_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "objFifo_out1_cons_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8>
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_2]]) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock_0"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objFifo_in1_buff_0"} : memref<64x64xi8>
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objFifo_in1_buff_1"} : memref<64x64xi8>
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "objFifo_in1_prod_lock_0"}
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "objFifo_in1_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_1]], DMA : 1)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_0]], DMA : 0)
// CHECK:           %[[VAL_19:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             %[[VAL_20:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_23:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 12 : i8
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_20]])
// CHECK:             scf.for %[[VAL_25:.*]] = %[[VAL_21]] to %[[VAL_23]] step %[[VAL_22]] {
// CHECK:               scf.for %[[VAL_26:.*]] = %[[VAL_21]] to %[[VAL_23]] step %[[VAL_22]] {
// CHECK:                 %[[VAL_27:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_25]], %[[VAL_26]]] : memref<64x64xi8>
// CHECK:                 %[[VAL_28:.*]] = arith.addi %[[VAL_27]], %[[VAL_24]] : i8
// CHECK:                 memref.store %[[VAL_28]], %[[VAL_11]]{{\[}}%[[VAL_25]], %[[VAL_26]]] : memref<64x64xi8>
// CHECK:               }
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_20]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_20]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.runtime_sequence(%[[VAL_29:.*]]: memref<61x56xi8>, %[[VAL_30:.*]]: memref<32xi8>, %[[VAL_31:.*]]: memref<64x64xi8>) {
// CHECK:             aiex.npu.dma_memcpy_nd(%[[VAL_29]][0, 0, 0, 0][1, 1, 1, 3416][0, 0, 0, 1]) {id = 0 : i64, metadata = @objFifo_in0_shim_alloc} : memref<61x56xi8>
// CHECK:             aiex.npu.dma_memcpy_nd(%[[VAL_31]][0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0_shim_alloc} : memref<64x64xi8>
// CHECK:             aiex.npu.dma_wait {symbol = @objFifo_out0_shim_alloc}
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @objFifo_in0_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @objFifo_out0_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_32:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_33:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_35:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<64x64xi8> offset = 0 len = 4096 sizes = [61, 56] strides = [56, 1] pad [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<64x64xi8> offset = 0 len = 4096 sizes = [61, 56] strides = [56, 1] pad [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb12:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_39:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_40:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_41:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
      %elem = aie.objectfifo.acquire @objFifo_in1(Consume) : memref<64x64xi8>
      %subview1_obj0 = aie.objectfifo.acquire @objFifo_out1(Produce) : memref<64x64xi8>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c12_i8 = arith.constant 12 : i8
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %elem[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c12_i8 : i8
          memref.store %1, %elem[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.objectfifo.release @objFifo_in1(Consume) [1]
      aie.objectfifo.release @objFifo_out1(Produce) [1]
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<61x56xi8>, %arg1: memref<32xi8>, %arg2: memref<64x64xi8>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 61, 56][0, 0, 56, 1]) {id = 0 : i64, metadata = @objFifo_in0} : memref<61x56xi8>
      aiex.npu.dma_memcpy_nd (%arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64, 1]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<64x64xi8>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }
}
