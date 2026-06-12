//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
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
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[4096, 0, 0, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_1 : memref<8xi32> = dense<[4096, 0, 0, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_2 : memref<8xi32> = dense<[4096, 524288, 0, 0, 0, 0, 0, 2168586048]>
    memref.global "private" constant @blockwrite_data_3 : memref<8xi32> = dense<[4096, 1572864, 0, 0, 0, 0, 0, 2168586049]>
    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      %w32_addr = arith.constant 119300 : i32
      %w32_val = arith.constant 2147483648 : i32
      aiex.npu.write32(%w32_addr, %w32_val) {column = 0 : i32, row = 0 : i32} : i32, i32
      %2 = memref.get_global @blockwrite_data_2 : memref<8xi32>
      aiex.npu.blockwrite(%2) {address = 655360 : ui32, column = 0 : i32, row = 1 : i32} : memref<8xi32>
      %w32_addr_1 = arith.constant 656900 : i32
      %w32_val_1 = arith.constant 0 : i32
      aiex.npu.write32(%w32_addr_1, %w32_val_1) {column = 0 : i32, row = 1 : i32} : i32, i32
      %3 = memref.get_global @blockwrite_data_3 : memref<8xi32>
      aiex.npu.blockwrite(%3) {address = 655392 : ui32, column = 0 : i32, row = 1 : i32} : memref<8xi32>
      %w32_addr_2 = arith.constant 656948 : i32
      %w32_val_2 = arith.constant 1 : i32
      aiex.npu.write32(%w32_addr_2, %w32_val_2) {column = 0 : i32, row = 1 : i32} : i32, i32
      %1 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      %mw_addr = arith.constant 119296 : i32
      %mw_val = arith.constant 256 : i32
      %mw_mask = arith.constant 3840 : i32
      aiex.npu.maskwrite32(%mw_addr, %mw_val, %mw_mask) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %w32_addr_3 = arith.constant 119316 : i32
      %w32_val_3 = arith.constant 1 : i32
      aiex.npu.write32(%w32_addr_3, %w32_val_3) {column = 0 : i32, row = 0 : i32} : i32, i32
      %column = arith.constant 0 : i32
      %row = arith.constant 0 : i32
      %direction = arith.constant 0 : i32
      %channel = arith.constant 0 : i32
      %column_num = arith.constant 1 : i32
      %row_num = arith.constant 1 : i32
      aiex.npu.sync(%column, %row, %direction, %channel, %column_num, %row_num) : i32, i32, i32, i32, i32, i32
    }
  }
}

