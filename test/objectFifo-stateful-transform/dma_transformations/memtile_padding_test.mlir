//===- memtile_padding_test.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objFifo_in1_buff_0"} : memref<64x64xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objFifo_in1_buff_1"} : memref<64x64xi8>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "objFifo_in1_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "objFifo_in1_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "objFifo_in1_prod_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "objFifo_in1_cons_lock_1"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objFifo_out1_cons_buff_0"} : memref<64x64xi8>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objFifo_out1_cons_buff_1"} : memref<64x64xi8>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "objFifo_out1_cons_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "objFifo_out1_cons_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_12]]) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_12]]) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8>
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_12]]) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_12]]) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_12]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_12]], DMA : 0, %[[VAL_1]], DMA : 1)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_0]], DMA : 0)
// CHECK:           %[[VAL_21:.*]] = aie.core(%[[VAL_12]]) {
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_25:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_26:.*]] = arith.constant 12 : i8
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             scf.for %[[VAL_27:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_24]] {
// CHECK:               scf.for %[[VAL_28:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_24]] {
// CHECK:                 %[[VAL_29:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_27]], %[[VAL_28]]] : memref<64x64xi8>
// CHECK:                 %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_26]] : i8
// CHECK:                 memref.store %[[VAL_30]], %[[VAL_13]]{{\[}}%[[VAL_27]], %[[VAL_28]]] : memref<64x64xi8>
// CHECK:               }
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_22]])
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_22]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.runtime_sequence(%[[VAL_31:.*]]: memref<61x56xi8>, %[[VAL_32:.*]]: memref<32xi8>, %[[VAL_33:.*]]: memref<64x64xi8>) {
// CHECK:             aiex.npu.dma_memcpy_nd(%[[VAL_31]][0, 0, 0, 0][1, 1, 1, 3416][0, 0, 0, 1]) {id = 0 : i64, metadata = @objFifo_in0_shim_alloc} : memref<61x56xi8>
// CHECK:             aiex.npu.dma_memcpy_nd(%[[VAL_33]][0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0_shim_alloc} : memref<64x64xi8>
// CHECK:             aiex.npu.dma_wait {symbol = @objFifo_out0_shim_alloc}
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @objFifo_in0_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           aie.shim_dma_allocation @objFifo_out0_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_34:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<64x64xi8> offset = 0 len = 3136)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<64x64xi8> offset = 0 len = 3136)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb8)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<64x64xi8> offset = 0 len = 3136)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<64x64xi8> offset = 3136 len = 960)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<64x64xi8> offset = 0 len = 3136)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<64x64xi8> offset = 3136 len = 960)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb8:
// CHECK:             %[[VAL_38:.*]] = aie.dma_start(S2MM, 1, ^bb9, ^bb11)
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb11:
// CHECK:             %[[VAL_39:.*]] = aie.dma_start(MM2S, 1, ^bb12, ^bb14)
// CHECK:           ^bb12:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<64x64xi8> offset = 0 len = 4096 sizes = [61, 56] strides = [56, 1] pad [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<64x64xi8> offset = 0 len = 4096 sizes = [61, 56] strides = [56, 1] pad [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb14:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = aie.mem(%[[VAL_12]]) {
// CHECK:             %[[VAL_41:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_42:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_41]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_41]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_43:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_41]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<64x64xi8> offset = 0 len = 4096)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_41]])
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
