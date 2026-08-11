//===- aie.mlir --------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Deterministic merge, proven by the ARRIVAL ORDER rather than by liveness.
//
// Three source tiles each send exactly one 256-byte packet (all packet id 0) to
// the same shim DMA. They merge at the shim stream switch on arbiter 0, which
// has three slaves -- one per source:
//
//   North : 2  <- tile(0,2), payload 1
//   East  : 2  <- tile(0,3), payload 2
//   East  : 0  <- tile(0,4), payload 3
//
// Left to arbitrate freely the hardware delivers 1, 3, 2 -- measured identical
// across 12 consecutive runs. That order is reproducible but nothing in the
// architecture guarantees it; it just falls out of path latency and arbiter
// state.
//
// The schedule below demands 2, 1, 3 instead: a permutation the hardware does
// NOT produce on its own. So the assertion in test.cpp can only pass if the
// arbiter actually honoured the programmed slot order. If deterministic merge
// regresses -- emission drops, wrong register, wrong slot order, hardware
// ignores it -- the output reverts to 1, 3, 2 and the test fails.
//
// This is deliberately NOT a deadlock test. Every packet is delivered either
// way and the destination BD takes all 768 bytes, so neither outcome can stall
// the stream or wedge the device. Each source produces exactly the one packet
// its slot grants.
//
// The switchbox ops are `--aie-create-pathfinder-flows` output with the
// attribute added and the aie.packet_flow ops removed, which leaves the
// configuration untouched by the router.

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0) deterministic_merge [<East : 2, 1>, <North : 2, 1>, <East : 0, 1>]
      %1 = aie.masterset(South : 2, %0)
      aie.packet_rules(East : 0) {
        aie.rule(31, 0, %0)
      }
      aie.packet_rules(East : 2) {
        aie.rule(31, 0, %0)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 2, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(East : 0, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_4 = aie.tile(0, 4)
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(East : 3, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<256xi8> 
    %prod0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "prod0"}
    %cons0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "cons0"}
    %buf1 = aie.buffer(%tile_0_3) {sym_name = "buf1"} : memref<256xi8> 
    %prod1 = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "prod1"}
    %cons1 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "cons1"}
    %buf2 = aie.buffer(%tile_0_4) {sym_name = "buf2"} : memref<256xi8> 
    %prod2 = aie.lock(%tile_0_4, 0) {init = 1 : i32, sym_name = "prod2"}
    %cons2 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "cons2"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c1_i8 = arith.constant 1 : i8
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%prod0, AcquireGreaterEqual, %c1_i32)
      scf.for %arg0 = %c0 to %c256 step %c1 {
        memref.store %c1_i8, %buf0[%arg0] : memref<256xi8>
      }
      aie.use_lock(%cons0, Release, %c1_i32)
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(MM2S, 0) [{
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%cons0, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buf0 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%prod0, Release, %c1_i32)
      }]
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c2_i8 = arith.constant 2 : i8
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%prod1, AcquireGreaterEqual, %c1_i32)
      scf.for %arg0 = %c0 to %c256 step %c1 {
        memref.store %c2_i8, %buf1[%arg0] : memref<256xi8>
      }
      aie.use_lock(%cons1, Release, %c1_i32)
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma(MM2S, 0) [{
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%cons1, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buf1 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%prod1, Release, %c1_i32)
      }]
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c3_i8 = arith.constant 3 : i8
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%prod2, AcquireGreaterEqual, %c1_i32)
      scf.for %arg0 = %c0 to %c256 step %c1 {
        memref.store %c3_i8, %buf2[%arg0] : memref<256xi8>
      }
      aie.use_lock(%cons2, Release, %c1_i32)
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma(MM2S, 0) [{
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%cons2, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buf2 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%prod2, Release, %c1_i32)
      }]
      aie.end
    }
    aie.shim_dma_allocation @out0(%shim_noc_tile_0_0, S2MM, 0)
    aie.runtime_sequence(%arg0: memref<768xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c768_i64 = arith.constant 768 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c768_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<768xi8>
      aiex.npu.dma_wait {symbol = @out0}
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 2, %0)
      aie.packet_rules(North : 2) {
        aie.rule(31, 0, %0)
      }
    }
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(West : 0, %0)
      %3 = aie.masterset(West : 2, %1)
      aie.packet_rules(North : 0) {
        aie.rule(31, 0, %0)
      }
      aie.packet_rules(North : 3) {
        aie.rule(31, 0, %1)
      }
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 0, %0)
      %3 = aie.masterset(South : 3, %1)
      aie.packet_rules(North : 0) {
        aie.rule(31, 0, %0)
      }
      aie.packet_rules(North : 3) {
        aie.rule(31, 0, %1)
      }
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 0, %0)
      %3 = aie.masterset(South : 3, %1)
      aie.packet_rules(North : 0) {
        aie.rule(31, 0, %0)
      }
      aie.packet_rules(North : 1) {
        aie.rule(31, 0, %1)
      }
    }
    %tile_1_3 = aie.tile(1, 3)
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 0, %1)
      %3 = aie.masterset(South : 1, %0)
      aie.packet_rules(North : 3) {
        aie.rule(31, 0, %1)
      }
      aie.packet_rules(West : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_1_4 = aie.tile(1, 4)
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 3, %0)
      aie.packet_rules(West : 3) {
        aie.rule(31, 0, %0)
      }
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
  }
}

