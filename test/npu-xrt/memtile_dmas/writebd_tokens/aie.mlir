//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {

    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    
    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_1, "TileControl" : 0>
      aie.packet_dest<%tile_0_0, "South" : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_0, "TileControl" : 0>
      aie.packet_dest<%tile_0_0, "South" : 0>
    }

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {

      // BD0, DMA_S2MM_0_Task_Queue
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %ap_argplus_c0 = arith.constant 0 : i32
      aiex.npu.address_patch(%ap_argplus_c0 : i32) {addr = 0x1d004 : ui32, arg_idx = 2 : i32}
      %mw_addr = arith.constant 119296 : i32
      %mw_val = arith.constant 512 : i32
      %mw_mask = arith.constant 3840 : i32
      aiex.npu.maskwrite32(%mw_addr, %mw_val, %mw_mask) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %w32_addr_1 = arith.constant 119300 : i32
      %w32_val_1 = arith.constant 2147483648 : i32
      aiex.npu.write32(%w32_addr_1, %w32_val_1) {column = 0 : i32, row = 0 : i32} : i32, i32

      // BD1, DMA_MM2S_0_Task_Queue
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %ap_argplus_c1 = arith.constant 0 : i32
      aiex.npu.address_patch(%ap_argplus_c1 : i32) {addr = 0x1d024 : ui32, arg_idx = 0 : i32}
      %w32_addr_2 = arith.constant 119316 : i32
      %w32_val_2 = arith.constant 1 : i32
      aiex.npu.write32(%w32_addr_2, %w32_val_2) {column = 0 : i32, row = 0 : i32} : i32, i32

      // BD0, DMA_S2MM_0_Start_Queue
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %mw_addr_3 = arith.constant 656896 : i32
      %mw_val_3 = arith.constant 256 : i32
      %mw_mask_3 = arith.constant 3840 : i32
      aiex.npu.maskwrite32(%mw_addr_3, %mw_val_3, %mw_mask_3) {column = 0 : i32, row = 1 : i32} : i32, i32, i32
      %w32_addr_4 = arith.constant 656900 : i32
      %w32_val_4 = arith.constant 2147483648 : i32
      aiex.npu.write32(%w32_addr_4, %w32_val_4) {column = 0 : i32, row = 1 : i32} : i32, i32

      // sync with the copy into memtile before starting copy out of memtile
      %column = arith.constant 0 : i32
      %row = arith.constant 1 : i32
      %direction = arith.constant 0 : i32
      %channel = arith.constant 0 : i32
      %column_num = arith.constant 1 : i32
      %row_num = arith.constant 1 : i32
      aiex.npu.sync(%column, %row, %direction, %channel, %column_num, %row_num) : i32, i32, i32, i32, i32, i32

      // BD1, DMA_MM2S_0_Start_Queue
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %w32_addr_5 = arith.constant 656948 : i32
      %w32_val_5 = arith.constant 1 : i32
      aiex.npu.write32(%w32_addr_5, %w32_val_5) {column = 0 : i32, row = 1 : i32} : i32, i32

      // sync with the copy out via shimdma
      %column_1 = arith.constant 0 : i32
      %row_1 = arith.constant 0 : i32
      %direction_1 = arith.constant 0 : i32
      %channel_1 = arith.constant 0 : i32
      %column_num_1 = arith.constant 1 : i32
      %row_num_1 = arith.constant 1 : i32
      aiex.npu.sync(%column_1, %row_1, %direction_1, %channel_1, %column_num_1, %row_num_1) : i32, i32, i32, i32, i32, i32
    }
  }
}
 