//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    }
    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_0 = arith.constant 0 : i32
      aiex.npu.address_patch(%cst_npu_0 : i32) {addr = 118788 : ui32, arg_idx = 2 : i32}
      %cst_npu_1 = arith.constant 119296 : i32
      %cst_npu_2 = arith.constant 512 : i32
      %cst_npu_3 = arith.constant 3840 : i32
      aiex.npu.maskwrite32(%cst_npu_1, %cst_npu_2, %cst_npu_3) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %cst_npu_4 = arith.constant 119300 : i32
      %cst_npu_5 = arith.constant 2147483648 : i32
      aiex.npu.write32(%cst_npu_4, %cst_npu_5) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_6 = arith.constant 0 : i32
      aiex.npu.address_patch(%cst_npu_6 : i32) {addr = 118820 : ui32, arg_idx = 0 : i32}
      %cst_npu_7 = arith.constant 119316 : i32
      %cst_npu_8 = arith.constant 1 : i32
      aiex.npu.write32(%cst_npu_7, %cst_npu_8) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 2 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_9 = arith.constant 2219520 : i32
      %cst_npu_10 = arith.constant 256 : i32
      %cst_npu_11 = arith.constant 3840 : i32
      aiex.npu.maskwrite32(%cst_npu_9, %cst_npu_10, %cst_npu_11) {column = 0 : i32, row = 2 : i32} : i32, i32, i32
      %cst_npu_12 = arith.constant 2219524 : i32
      %cst_npu_13 = arith.constant 2147483648 : i32
      aiex.npu.write32(%cst_npu_12, %cst_npu_13) {column = 0 : i32, row = 2 : i32} : i32, i32
      %cst_npu_14 = arith.constant 0 : i32
      %cst_npu_15 = arith.constant 2 : i32
      %cst_npu_16 = arith.constant 0 : i32
      %cst_npu_17 = arith.constant 0 : i32
      %cst_npu_18 = arith.constant 1 : i32
      %cst_npu_19 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_14, %cst_npu_15, %cst_npu_16, %cst_npu_17, %cst_npu_18, %cst_npu_19) : i32, i32, i32, i32, i32, i32
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 2 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_20 = arith.constant 2219540 : i32
      %cst_npu_21 = arith.constant 1 : i32
      aiex.npu.write32(%cst_npu_20, %cst_npu_21) {column = 0 : i32, row = 2 : i32} : i32, i32
      %cst_npu_22 = arith.constant 0 : i32
      %cst_npu_23 = arith.constant 0 : i32
      %cst_npu_24 = arith.constant 0 : i32
      %cst_npu_25 = arith.constant 0 : i32
      %cst_npu_26 = arith.constant 1 : i32
      %cst_npu_27 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_22, %cst_npu_23, %cst_npu_24, %cst_npu_25, %cst_npu_26, %cst_npu_27) : i32, i32, i32, i32, i32, i32
    }
  }
}
