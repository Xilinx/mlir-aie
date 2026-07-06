//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %tile_0_1 = aie.tile(0, 1)
    %in_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "in_cons_buff_0"} : memref<4096xi32> 
    %in_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "in_cons_prod_lock"}
    %in_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    aie.shim_dma_allocation @in (%tile_0_0, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_0 = arith.constant 0 : i32
      aiex.npu.address_patch(%cst_npu_0 : i32) {addr = 118788 : ui32, arg_idx = 2 : i32}
      %cst_npu_1 = arith.constant 119300 : i32
      %cst_npu_2 = arith.constant 2147483648 : i32
      aiex.npu.write32(%cst_npu_1, %cst_npu_2) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 1 : i32, lock_acq_id = 64 : i32, lock_acq_val = 127 : i32, lock_rel_id = 65 : i32, lock_rel_val = 1 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 1 : i32, valid_bd = 1 : i32}
      %cst_npu_3 = arith.constant 656900 : i32
      %cst_npu_4 = arith.constant 0 : i32
      aiex.npu.write32(%cst_npu_3, %cst_npu_4) {column = 0 : i32, row = 1 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 1 : i32, lock_acq_id = 65 : i32, lock_acq_val = 127 : i32, lock_rel_id = 64 : i32, lock_rel_val = 1 : i32, next_bd = 1 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 1 : i32, valid_bd = 1 : i32}
      %cst_npu_5 = arith.constant 656948 : i32
      %cst_npu_6 = arith.constant 1 : i32
      aiex.npu.write32(%cst_npu_5, %cst_npu_6) {column = 0 : i32, row = 1 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_7 = arith.constant 0 : i32
      aiex.npu.address_patch(%cst_npu_7 : i32) {addr = 118820 : ui32, arg_idx = 0 : i32}
      %cst_npu_8 = arith.constant 119296 : i32
      %cst_npu_9 = arith.constant 0x100 : i32
      %cst_npu_10 = arith.constant 0x00000F00 : i32
      aiex.npu.maskwrite32(%cst_npu_8, %cst_npu_9, %cst_npu_10) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %cst_npu_11 = arith.constant 119316 : i32
      %cst_npu_12 = arith.constant 1 : i32
      aiex.npu.write32(%cst_npu_11, %cst_npu_12) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_13 = arith.constant 0 : i32
      %cst_npu_14 = arith.constant 0 : i32
      %cst_npu_15 = arith.constant 0 : i32
      %cst_npu_16 = arith.constant 0 : i32
      %cst_npu_17 = arith.constant 1 : i32
      %cst_npu_18 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_13, %cst_npu_14, %cst_npu_15, %cst_npu_16, %cst_npu_17, %cst_npu_18) : i32, i32, i32, i32, i32, i32
    }
  }
}
 
