//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %tile_0_2 = aie.tile(0, 2)
    %in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 0 : i32, sym_name = "in_cons_buff_0"} : memref<4096xi32> 
    %in_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "in_cons_prod_lock"}
    %in_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.shim_dma_allocation @in(%shim_noc_tile_0_0, MM2S, 0)
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[4096, 0, 0, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_1 : memref<8xi32> = dense<[4096, 0, 0, 0, -2147483648, 0, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_2 : memref<6xi32> = dense<[4096, 0, 0, 0, 0, 33828896]>
    memref.global "private" constant @blockwrite_data_3 : memref<6xi32> = dense<[4096, 0, 0, 0, 0, 33820705]>
    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      %cst_npu_0 = arith.constant 0 : i32
      aiex.npu.address_patch(%cst_npu_0 : i32) {addr = 118788 : ui32, arg_idx = 2 : i32}
      %cst_npu_1 = arith.constant 119300 : i32
      %cst_npu_2 = arith.constant 2147483648 : i32
      aiex.npu.write32(%cst_npu_1, %cst_npu_2) {column = 0 : i32, row = 0 : i32} : i32, i32
      %1 = memref.get_global @blockwrite_data_2 : memref<6xi32>
      aiex.npu.blockwrite(%1) {address = 2215936 : ui32, column = 0 : i32, row = 2 : i32} : memref<6xi32>
      %cst_npu_3 = arith.constant 2219524 : i32
      %cst_npu_4 = arith.constant 0 : i32
      aiex.npu.write32(%cst_npu_3, %cst_npu_4) {column = 0 : i32, row = 2 : i32} : i32, i32
      %2 = memref.get_global @blockwrite_data_3 : memref<6xi32>
      aiex.npu.blockwrite(%2) {address = 2215968 : ui32, column = 0 : i32, row = 2 : i32} : memref<6xi32>
      %cst_npu_5 = arith.constant 2219540 : i32
      %cst_npu_6 = arith.constant 1 : i32
      aiex.npu.write32(%cst_npu_5, %cst_npu_6) {column = 0 : i32, row = 2 : i32} : i32, i32
      %3 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%3) {address = 118816 : ui32} : memref<8xi32>
      %cst_npu_7 = arith.constant 0 : i32
      aiex.npu.address_patch(%cst_npu_7 : i32) {addr = 118820 : ui32, arg_idx = 0 : i32}
      %cst_npu_8 = arith.constant 119296 : i32
      %cst_npu_9 = arith.constant 256 : i32
      %cst_npu_10 = arith.constant 3840 : i32
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
