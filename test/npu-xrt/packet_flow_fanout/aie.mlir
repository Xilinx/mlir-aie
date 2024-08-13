//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022-2024 Advanced Micro Devices, Inc. or its affiliates
// Copyright (C) 2020-2022, Xilinx Inc.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    memref.global "public" @objFifo_in0 : memref<128x64xi8>
    memref.global "public" @objFifo_out0 : memref<128x64xi8>
    memref.global "public" @objFifo_out1 : memref<128x64xi8>

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    %objFifo_core02_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_core02_cons_buff_0"} : memref<64x64xi8>
    %objFifo_core02_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_core02_cons_buff_1"} : memref<64x64xi8>
    %objFifo_core02_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_core02_buff_0"} : memref<64x64xi8>
    %objFifo_core02_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_core02_buff_1"} : memref<64x64xi8>

    %objFifo_core02_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "objFifo_core02_cons_prod_lock"}
    %objFifo_core02_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_core02_cons_cons_lock"}
    %objFifo_core02_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "objFifo_core02_prod_lock"}
    %objFifo_core02_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_core02_cons_lock"}

    %objFifo_core03_cons_buff_0 = aie.buffer(%tile_0_3) {sym_name = "objFifo_core03_cons_buff_0"} : memref<64x64xi8>
    %objFifo_core03_cons_buff_1 = aie.buffer(%tile_0_3) {sym_name = "objFifo_core03_cons_buff_1"} : memref<64x64xi8>
    %objFifo_core03_buff_0 = aie.buffer(%tile_0_3) {sym_name = "objFifo_core03_buff_0"} : memref<64x64xi8>
    %objFifo_core03_buff_1 = aie.buffer(%tile_0_3) {sym_name = "objFifo_core03_buff_1"} : memref<64x64xi8>

    %objFifo_core03_cons_prod_lock = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "objFifo_core03_cons_prod_lock"}
    %objFifo_core03_cons_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "objFifo_core03_cons_cons_lock"}
    %objFifo_core03_prod_lock = aie.lock(%tile_0_3, 2) {init = 1 : i32, sym_name = "objFifo_core03_prod_lock"}
    %objFifo_core03_cons_lock = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "objFifo_core03_cons_lock"}

    aie.packet_flow(0) {
      aie.packet_source<%tile_0_1, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_0_1, DMA : 1>
    }
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_1, DMA : 1>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
    aie.packet_flow(3) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, DMA : 0>
    }

    aie.packet_flow(4) {
      aie.packet_source<%tile_0_1, DMA : 2>
      aie.packet_dest<%tile_0_3, DMA : 0>
    }
    aie.packet_flow(5) {
      aie.packet_source<%tile_0_3, DMA : 0>
      aie.packet_dest<%tile_0_1, DMA : 3>
    }
    aie.packet_flow(6) {
      aie.packet_source<%tile_0_1, DMA : 3>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
    aie.packet_flow(7) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, DMA : 2>
    }

    %core_0_2 = aie.core(%tile_0_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12_i8 = arith.constant 12 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%objFifo_core02_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%objFifo_core02_prod_lock, AcquireGreaterEqual, 1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %objFifo_core02_cons_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c12_i8 : i8
          memref.store %1, %objFifo_core02_buff_0[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.use_lock(%objFifo_core02_cons_prod_lock, Release, 1)
      aie.use_lock(%objFifo_core02_cons_lock, Release, 1)
      aie.end
    }

    %core_0_3 = aie.core(%tile_0_3) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12_i8 = arith.constant 12 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%objFifo_core03_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%objFifo_core03_prod_lock, AcquireGreaterEqual, 1)
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %objFifo_core03_cons_buff_0[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c12_i8 : i8
          memref.store %1, %objFifo_core03_buff_0[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.use_lock(%objFifo_core03_cons_prod_lock, Release, 1)
      aie.use_lock(%objFifo_core03_cons_lock, Release, 1)
      aie.end
    }

    aie.shim_dma_allocation @objFifo_in0(MM2S, 0, 0)

    aiex.runtime_sequence(%arg0: memref<128x64xi8>, %arg1: memref<32xi8>, %arg2: memref<128x64xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c64_i64 = arith.constant 64 : i64
      // Packet-flow fanout happening at shim dma channel @objFifo_in0, where packet id 3 and 7 go to tile_0_1's S2MM DMA channel 0 and 2, respectively
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64], packet = <pkt_id = 3, pkt_type = 0>) {id = 0 : i64, metadata = @objFifo_in0} : memref<128x64xi8>
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[%c0_i64, %c0_i64, %c64_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64], packet = <pkt_id = 7, pkt_type = 0>) {id = 1 : i64, metadata = @objFifo_in0} : memref<128x64xi8>
      aiex.npu.dma_memcpy_nd (0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64]) {id = 2 : i64, metadata = @objFifo_out0, issue_token = true} : memref<128x64xi8>
      aiex.npu.dma_memcpy_nd (0, 0, %arg2[%c0_i64, %c0_i64, %c64_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64]) {id = 3 : i64, metadata = @objFifo_out1, issue_token = true} : memref<128x64xi8>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
      aiex.npu.dma_wait { symbol = @objFifo_out1 }
    }

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in0_cons_buff_0"} : memref<64x64xi8>
      %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in0_cons_buff_1"} : memref<64x64xi8>
      %objFifo_out0_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out0_buff_0"} : memref<64x64xi8>
      %objFifo_out0_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out0_buff_1"} : memref<64x64xi8>
      %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
      %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
      %objFifo_out0_prod_lock = aie.lock(%tile_0_1, 2) {init = 1 : i32, sym_name = "objFifo_out0_prod_lock"}
      %objFifo_out0_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}
      
      %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8>
      %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8>
      %objFifo_out1_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8>
      %objFifo_out1_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8>
      %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_1, 4) {init = 1 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
      %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_1, 5) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
      %objFifo_out1_prod_lock = aie.lock(%tile_0_1, 6) {init = 1 : i32, sym_name = "objFifo_out1_prod_lock"}
      %objFifo_out1_cons_lock = aie.lock(%tile_0_1, 7) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8>)
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      }]
      %2 = aie.dma(MM2S, 1) [{
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      }]
      %3 = aie.dma(S2MM, 1) [{
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>)
        aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      }]
      
      %4 = aie.dma(S2MM, 2) [{
        aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<64x64xi8>)
        aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      }]
      %5 = aie.dma(MM2S, 2) [{
        aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<64x64xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
        aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
      }]
      %6 = aie.dma(MM2S, 3) [{
        aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out1_buff_0 : memref<64x64xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
        aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      }]
      %7 = aie.dma(S2MM, 3) [{
        aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out1_buff_0 : memref<64x64xi8>)
        aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
      }]
      aie.end
    }

    aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)
    aie.shim_dma_allocation @objFifo_out1(S2MM, 1, 0)

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_core02_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_core02_cons_buff_0 : memref<64x64xi8>)
        aie.use_lock(%objFifo_core02_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_core02_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_core02_buff_0 : memref<64x64xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%objFifo_core02_prod_lock, Release, 1)
      }]
      aie.end
    }

    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_core03_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_core03_cons_buff_0 : memref<64x64xi8>)
        aie.use_lock(%objFifo_core03_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_core03_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_core03_buff_0 : memref<64x64xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
        aie.use_lock(%objFifo_core03_prod_lock, Release, 1)
      }]
      aie.end
    }
  }
}
