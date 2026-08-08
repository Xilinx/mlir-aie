//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Overlapping packet rules misroute silently (Xilinx/mlir-aie#437).
//
// Two packet streams leave shim DMA MM2S 0 and fan apart at the memtile
// switchbox: id 10 -> memtile S2MM 0 (buf_a), id 14 -> memtile S2MM 1 (buf_b).
// The routing below is verbatim what --aie-create-pathfinder-flows emits for
// the equivalent flow-level design, except for the memtile South:4 rules, which
// are hand-edited to the overlapping pair from the issue:
//
//   aie.rule(26, 10, %0)   mask 0b11010 val 0b01010  -> masterset DMA:0
//   aie.rule(24,  8, %1)   mask 0b11000 val 0b01000  -> masterset DMA:1
//
// Id 14 (0b01110) matches both: 14 & 26 == 10 and 14 & 24 == 8. The switch takes
// the first match, so id 14 lands in buf_a and buf_b is never written. See
// README.md for what makes this falsifiable.

module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)

    // -1 is the sentinel the host looks for: buf_b keeps it when id 14 is stolen.
    %buf_a = aie.buffer(%t01) { initial_value = dense<-1> : tensor<8xi32>, sym_name = "buf_a" } : memref<8xi32>
    %buf_b = aie.buffer(%t01) { initial_value = dense<-1> : tensor<8xi32>, sym_name = "buf_b" } : memref<8xi32>

    // a_arrived gates the buf_a readback on data actually arriving; b_go gates
    // the buf_b readback on the runtime sequence, *not* on data, so the readback
    // still completes when id 14 never shows up. No path can deadlock.
    %a_arrived = aie.lock(%t01, 0) {init = 0 : i32, sym_name = "a_arrived"}
    %b_go = aie.lock(%t01, 1) {init = 0 : i32, sym_name = "b_go"}
    // Every BD needs one acquire and one release, so the BDs that should not
    // gate at all take from a pre-filled lock and release into one nobody reads.
    %ready = aie.lock(%t01, 2) {init = 16 : i32, sym_name = "ready"}
    %unused = aie.lock(%t01, 3) {init = 0 : i32, sym_name = "unused"}

    %switchbox_0_0 = aie.switchbox(%t00) {
      aie.connect<North : 2, South : 2>
      aie.connect<North : 3, South : 3>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 4, %0)
      // Both ids share this master, so one relaxed rule covering {10, 14} is
      // correct here -- 10 & 27 == 14 & 27 == 10.
      aie.packet_rules(South : 3) {
        aie.rule(27, 10, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%t00) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_0_1 = aie.switchbox(%t01) {
      aie.connect<DMA : 0, South : 2>
      aie.connect<DMA : 1, South : 3>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %0)
      %3 = aie.masterset(DMA : 1, %1)
      // The router emits rule(31, 14, %1) / rule(31, 10, %0) here. Relaxing the
      // masks is what breaks it.
      aie.packet_rules(South : 4) {
        aie.rule(26, 10, %0)
        aie.rule(24, 8, %1)
      }
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%t00 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%t01 : Core, %switchbox_0_1 : Core)
    aie.wire(%t01 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)

    // Memtile BD ids are partitioned by channel parity: even channels take
    // BD < 24, odd channels BD >= 24.
    %memtile_dma_0_1 = aie.memtile_dma(%t01) {
      // S2MM 0: takes from `ready`, so it accepts one packet or two -- under the
      // bug it receives both streams -- without ever stalling.
      %s0 = aie.dma_start(S2MM, 0, ^bd0, ^dma1)
    ^bd0:
      %c1_0 = arith.constant 1 : i32
      aie.use_lock(%ready, AcquireGreaterEqual, %c1_0)
      aie.dma_bd(%buf_a : memref<8xi32>) {bd_id = 0 : i32}
      aie.use_lock(%a_arrived, Release, %c1_0)
      aie.next_bd ^bd0
    ^dma1:
      %s1 = aie.dma_start(MM2S, 0, ^bd1, ^dma2)
    ^bd1:
      %c1_1 = arith.constant 1 : i32
      aie.use_lock(%a_arrived, AcquireGreaterEqual, %c1_1)
      aie.dma_bd(%buf_a : memref<8xi32>) {bd_id = 1 : i32}
      aie.use_lock(%unused, Release, %c1_1)
      aie.next_bd ^bd1
    ^dma2:
      // S2MM 1 gates on nothing either: under the bug it simply never fires.
      %s2 = aie.dma_start(S2MM, 1, ^bd2, ^dma3)
    ^bd2:
      %c1_2 = arith.constant 1 : i32
      aie.use_lock(%ready, AcquireGreaterEqual, %c1_2)
      aie.dma_bd(%buf_b : memref<8xi32>) {bd_id = 24 : i32}
      aie.use_lock(%unused, Release, %c1_2)
      aie.next_bd ^bd2
    ^dma3:
      // Gated on the runtime sequence, not on data, so the readback still
      // completes when packet id 14 never arrives.
      %s3 = aie.dma_start(MM2S, 1, ^bd3, ^end)
    ^bd3:
      %c1_3 = arith.constant 1 : i32
      aie.use_lock(%b_go, AcquireGreaterEqual, %c1_3)
      aie.dma_bd(%buf_b : memref<8xi32>) {bd_id = 25 : i32}
      aie.use_lock(%unused, Release, %c1_3)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }

    aie.shim_dma_allocation @in (%t00, MM2S, 0)
    aie.shim_dma_allocation @outa (%t00, S2MM, 0)
    aie.shim_dma_allocation @outb (%t00, S2MM, 1)

    aie.runtime_sequence @seq(%x: memref<8xi32>, %y: memref<8xi32>, %oa: memref<8xi32>, %ob: memref<8xi32>) {
      // Send y (id 14) before x (id 10). Both share one wire, so y's memtile
      // write retires first; once path A has round-tripped, buf_b has settled.
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1], packet = <pkt_id = 14, pkt_type = 0>) {id = 0 : i64, metadata = @in} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1], packet = <pkt_id = 10, pkt_type = 0>) {id = 1 : i64, metadata = @in} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%oa[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @outa} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @outa}
      aiex.set_lock(%b_go, 1)
      aiex.npu.dma_memcpy_nd(%ob[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 3 : i64, issue_token = true, metadata = @outb} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @outb}
    }
  }
}
