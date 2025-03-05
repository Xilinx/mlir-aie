//===- aie_mul.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

module {
    aie.device(npu1_1col) {
        memref.global "public" @objFifo_in0 : memref<256xi8>
        memref.global "public" @objFifo_out0 : memref<256xi8>

        memref.global "public" @objFifo_in1 : memref<256xi8>
        memref.global "public" @objFifo_out1 : memref<256xi8>

        func.func private @add(memref<256xi8>, memref<256xi8>)
        func.func private @mul(memref<256xi8>, memref<256xi8>)

        %ShimTile_0_0 = aie.tile(0, 0)
        %MemTile_0_1 = aie.tile(0, 1)
        %CT_0_2 = aie.tile(0, 2)
        %CT_0_3 = aie.tile(0, 3)

        // core_0_2:
        %objFifo_core02_cons_buff_0 = aie.buffer(%CT_0_2) {sym_name = "objFifo_core02_cons_buff_0"} : memref<256xi8>
        %objFifo_core02_buff_0 = aie.buffer(%CT_0_2) {sym_name = "objFifo_core02_buff_0"} : memref<256xi8>

        %objFifo_core02_cons_prod_lock = aie.lock(%CT_0_2, 0) {init = 1 : i32, sym_name = "objFifo_core02_cons_prod_lock"}
        %objFifo_core02_cons_cons_lock = aie.lock(%CT_0_2, 1) {init = 0 : i32, sym_name = "objFifo_core02_cons_cons_lock"}
        %objFifo_core02_prod_lock = aie.lock(%CT_0_2, 4) {init = 1 : i32, sym_name = "objFifo_core02_prod_lock"}
        %objFifo_core02_cons_lock = aie.lock(%CT_0_2, 5) {init = 0 : i32, sym_name = "objFifo_core02_cons_lock"}

        // core_0_3: 
        %objFifo_core03_cons_buff_0 = aie.buffer(%CT_0_3) {sym_name = "objFifo_core03_cons_buff_0"} : memref<256xi8>
        %objFifo_core03_buff_0 = aie.buffer(%CT_0_3) {sym_name = "objFifo_core03_buff_0"} : memref<256xi8>

        %objFifo_core03_cons_prod_lock = aie.lock(%CT_0_3, 0) {init = 1 : i32, sym_name = "objFifo_core03_cons_prod_lock"}
        %objFifo_core03_cons_cons_lock = aie.lock(%CT_0_3, 1) {init = 0 : i32, sym_name = "objFifo_core03_cons_cons_lock"}
        %objFifo_core03_prod_lock = aie.lock(%CT_0_3, 4) {init = 1 : i32, sym_name = "objFifo_core03_prod_lock"}
        %objFifo_core03_cons_lock = aie.lock(%CT_0_3, 5) {init = 0 : i32, sym_name = "objFifo_core03_cons_lock"}

        // Add
        aie.packet_flow(0) {
            aie.packet_source<%ShimTile_0_0, DMA : 0>
            aie.packet_dest<%MemTile_0_1, DMA : 0>
        } {keep_pkt_header = true} 

        aie.packet_flow(1) {
            aie.packet_source<%ShimTile_0_0, DMA : 0>
            aie.packet_dest<%MemTile_0_1, DMA : 0>
        } {keep_pkt_header = true}

        aie.packet_flow(2) {
            aie.packet_source<%MemTile_0_1, DMA : 2>
            aie.packet_dest<%ShimTile_0_0, DMA : 0>
        }
        aie.packet_flow(0) {
            aie.packet_source<%MemTile_0_1, DMA : 0>
            aie.packet_dest<%CT_0_2, DMA : 0>
        }
        aie.packet_flow(4) {
            aie.packet_source<%CT_0_2, DMA : 0>
            aie.packet_dest<%MemTile_0_1, DMA : 2>
        }
        aie.packet_flow(1) {
            aie.packet_source<%MemTile_0_1, DMA : 0>
            aie.packet_dest<%CT_0_3, DMA : 0>
        }
        aie.packet_flow(6) {
            aie.packet_source<%CT_0_3, DMA : 0>
            aie.packet_dest<%MemTile_0_1, DMA : 2>
        }

        
        %core_0_2 = aie.core(%CT_0_2) {
            %c0 = arith.constant 0 : index
            %c4294967295 = arith.constant 4294967295 : index
            %c1 = arith.constant 1 : index
            scf.for %arg0 = %c0 to %c4294967295 step %c1 {
                aie.use_lock(%objFifo_core02_cons_cons_lock, AcquireGreaterEqual, 1)
                aie.use_lock(%objFifo_core02_prod_lock, AcquireGreaterEqual, 1)
                func.call @add(%objFifo_core02_cons_buff_0,  %objFifo_core02_buff_0) : (memref<256xi8>, memref<256xi8>) -> ()
                aie.use_lock(%objFifo_core02_cons_prod_lock, Release, 1)
                aie.use_lock(%objFifo_core02_cons_lock, Release, 1)
            }
            aie.end
        } {link_with = "add_mul.o"}

        %mem_0_2 = aie.mem(%CT_0_2) {
            %0 = aie.dma(S2MM, 0) [
                {
                    aie.use_lock(%objFifo_core02_cons_prod_lock, AcquireGreaterEqual, 1)
                    aie.dma_bd(%objFifo_core02_cons_buff_0 : memref<256xi8>)
                    aie.use_lock(%objFifo_core02_cons_cons_lock, Release, 1)
                }
            ]
            %1 = aie.dma(MM2S, 0) [{
                aie.use_lock(%objFifo_core02_cons_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%objFifo_core02_buff_0 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
                aie.use_lock(%objFifo_core02_prod_lock, Release, 1)
            }]
            aie.end
        }

        %core_0_3 = aie.core(%CT_0_3) {
            %c0 = arith.constant 0 : index
            %c4294967295 = arith.constant 4294967295 : index
            %c1 = arith.constant 1 : index
            scf.for %arg0 = %c0 to %c4294967295 step %c1 {
                aie.use_lock(%objFifo_core03_cons_cons_lock, AcquireGreaterEqual, 1)
                aie.use_lock(%objFifo_core03_prod_lock, AcquireGreaterEqual, 1)
                func.call @mul(%objFifo_core03_cons_buff_0, %objFifo_core03_buff_0) : (memref<256xi8>, memref<256xi8>) -> ()
                aie.use_lock(%objFifo_core03_cons_prod_lock, Release, 1)
                aie.use_lock(%objFifo_core03_cons_lock, Release, 1)
            }
            aie.end
                
        } {link_with = "add_mul.o"}

        %mem_0_3 = aie.mem(%CT_0_3) {
            %0 = aie.dma(S2MM, 0) [{
                aie.use_lock(%objFifo_core03_cons_prod_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%objFifo_core03_cons_buff_0 : memref<256xi8>)
                aie.use_lock(%objFifo_core03_cons_cons_lock, Release, 1)
            }]
            %1 = aie.dma(MM2S, 0) [{
                aie.use_lock(%objFifo_core03_cons_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%objFifo_core03_buff_0 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
                aie.use_lock(%objFifo_core03_prod_lock, Release, 1)
            }]
            aie.end
        }


        aie.shim_dma_allocation @objFifo_in1(MM2S, 0, 0)
        aiex.runtime_sequence(%arg0: memref<256xi8>, %arg2: memref<256xi8>) {
            aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1], packet = <pkt_id = 1, pkt_type = 0>) {id = 0 : i64, metadata = @objFifo_in1} : memref<256xi8>
            aiex.npu.dma_memcpy_nd (%arg2[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @objFifo_out1, issue_token = true} : memref<256xi8>
            aiex.npu.dma_wait { symbol = @objFifo_out1 }
        }
        aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)
        aie.shim_dma_allocation @objFifo_out1(S2MM, 0, 0)

        %memtile_dma_0_1 = aie.memtile_dma(%MemTile_0_1) {
            %objFifo_in0_cons_buff_0 = aie.buffer(%MemTile_0_1) {sym_name = "objFifo_in0_cons_buff_0"} : memref<260xi8>
            %objFifo_out0_buff_0 = aie.buffer(%MemTile_0_1) {sym_name = "objFifo_out0_buff_0"} : memref<256xi8>

            %objFifo_in0_cons_prod_lock = aie.lock(%MemTile_0_1, 0) {init = 1 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
            %objFifo_in0_cons_cons_lock = aie.lock(%MemTile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
            %objFifo_out0_prod_lock = aie.lock(%MemTile_0_1, 4) {init = 1 : i32, sym_name = "objFifo_out0_prod_lock"}
            %objFifo_out0_cons_lock = aie.lock(%MemTile_0_1, 5) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}

            %0 = aie.dma(S2MM, 0) [{
                aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<260xi8>)
                aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
            }]
            %1 = aie.dma(MM2S, 0) [{
                aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<260xi8>) 
                aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
            }]
            %2 = aie.dma(MM2S, 2) [{
                aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%objFifo_out0_buff_0 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
                aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
            }]
            %3 = aie.dma(S2MM, 2) [{
                aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%objFifo_out0_buff_0 : memref<256xi8>)
                aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
            }]

            aie.end
        }
    }
}
