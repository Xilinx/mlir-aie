//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Relaxed packet-rule masks on hardware.
//
// Four packet ids share one shim source. Ids 0, 3 and 4 go to memtile S2MM 0
// and id 1 to memtile S2MM 1, which makes --aie-create-pathfinder-flows emit a
// two-cube cover for {0, 3, 4} at the memtile slave port:
//
//   aie.rule(31, 1, %1)   exact,   id 1       -> DMA:1
//   aie.rule(27, 0, %0)   relaxed, ids {0, 4} -> DMA:0
//   aie.rule(31, 3, %0)   exact,   id 3       -> DMA:0
//
// A single cube enclosing {0, 3, 4} would be {0..7} and would swallow id 1, so
// the cover has to split. The relaxed rule sits one bit away from id 1
// (1 & 27 == 1, not 0). Nothing here is hand-written: the point is to run the
// router's own mask minimisation on silicon. See README.md.

module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)

    aie.packet_flow(0) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 1> }

    // Readback is plain circuit-switched: only the inbound direction is under test.
    aie.flow(%t01, DMA : 0, %t00, DMA : 0)
    aie.flow(%t01, DMA : 1, %t00, DMA : 1)

    // -1 is the sentinel: a buffer no packet reached still reads as -1 rather
    // than as a plausible payload.
    %a0 = aie.buffer(%t01) { initial_value = dense<-1> : tensor<8xi32>, sym_name = "a0" } : memref<8xi32>
    %a1 = aie.buffer(%t01) { initial_value = dense<-1> : tensor<8xi32>, sym_name = "a1" } : memref<8xi32>
    %a2 = aie.buffer(%t01) { initial_value = dense<-1> : tensor<8xi32>, sym_name = "a2" } : memref<8xi32>
    %b0 = aie.buffer(%t01) { initial_value = dense<-1> : tensor<8xi32>, sym_name = "b0" } : memref<8xi32>

    // Receives never gate on a lock, so a misrouted id cannot stall a channel;
    // the readbacks gate on the runtime sequence rather than on data arriving,
    // so a buffer that got nothing still reads back as its sentinel. No routing
    // outcome can deadlock.
    %ready = aie.lock(%t01, 0) {init = 32 : i32, sym_name = "ready"}
    %unused = aie.lock(%t01, 1) {init = 0 : i32, sym_name = "unused"}
    %go_a = aie.lock(%t01, 2) {init = 0 : i32, sym_name = "go_a"}
    %go_b = aie.lock(%t01, 3) {init = 0 : i32, sym_name = "go_b"}

    // Memtile BD ids are partitioned by channel parity: even channels take
    // BD < 24, odd channels BD >= 24.
    %memtile_dma_0_1 = aie.memtile_dma(%t01) {
      // Ids 0, 3, 4 arrive here in send order and fill a0, a1, a2.
      %s0 = aie.dma_start(S2MM, 0, ^bd0, ^dma1)
    ^bd0:
      %c1_0 = arith.constant 1 : i32
      aie.use_lock(%ready, AcquireGreaterEqual, %c1_0)
      aie.dma_bd(%a0 : memref<8xi32>) {bd_id = 0 : i32}
      aie.use_lock(%unused, Release, %c1_0)
      aie.next_bd ^bd1
    ^bd1:
      %c1_1 = arith.constant 1 : i32
      aie.use_lock(%ready, AcquireGreaterEqual, %c1_1)
      aie.dma_bd(%a1 : memref<8xi32>) {bd_id = 1 : i32}
      aie.use_lock(%unused, Release, %c1_1)
      aie.next_bd ^bd2
    ^bd2:
      %c1_2 = arith.constant 1 : i32
      aie.use_lock(%ready, AcquireGreaterEqual, %c1_2)
      aie.dma_bd(%a2 : memref<8xi32>) {bd_id = 2 : i32}
      aie.use_lock(%unused, Release, %c1_2)
      aie.next_bd ^bd0
    ^dma1:
      // Readback of a0, a1, a2 back-to-back into one shim receive.
      %s1 = aie.dma_start(MM2S, 0, ^bd3, ^dma2)
    ^bd3:
      %c1_3 = arith.constant 1 : i32
      aie.use_lock(%go_a, AcquireGreaterEqual, %c1_3)
      aie.dma_bd(%a0 : memref<8xi32>) {bd_id = 3 : i32}
      aie.use_lock(%unused, Release, %c1_3)
      aie.next_bd ^bd4
    ^bd4:
      %c1_4 = arith.constant 1 : i32
      aie.use_lock(%go_a, AcquireGreaterEqual, %c1_4)
      aie.dma_bd(%a1 : memref<8xi32>) {bd_id = 4 : i32}
      aie.use_lock(%unused, Release, %c1_4)
      aie.next_bd ^bd5
    ^bd5:
      %c1_5 = arith.constant 1 : i32
      aie.use_lock(%go_a, AcquireGreaterEqual, %c1_5)
      aie.dma_bd(%a2 : memref<8xi32>) {bd_id = 5 : i32}
      aie.use_lock(%unused, Release, %c1_5)
      aie.next_bd ^bd3
    ^dma2:
      %s2 = aie.dma_start(S2MM, 1, ^bd24, ^dma3)
    ^bd24:
      %c1_24 = arith.constant 1 : i32
      aie.use_lock(%ready, AcquireGreaterEqual, %c1_24)
      aie.dma_bd(%b0 : memref<8xi32>) {bd_id = 24 : i32}
      aie.use_lock(%unused, Release, %c1_24)
      aie.next_bd ^bd24
    ^dma3:
      %s3 = aie.dma_start(MM2S, 1, ^bd25, ^end)
    ^bd25:
      %c1_25 = arith.constant 1 : i32
      aie.use_lock(%go_b, AcquireGreaterEqual, %c1_25)
      aie.dma_bd(%b0 : memref<8xi32>) {bd_id = 25 : i32}
      aie.use_lock(%unused, Release, %c1_25)
      aie.next_bd ^bd25
    ^end:
      aie.end
    }

    aie.shim_dma_allocation @in (%t00, MM2S, 0)
    aie.shim_dma_allocation @outa (%t00, S2MM, 0)
    aie.shim_dma_allocation @outb (%t00, S2MM, 1)

    aie.runtime_sequence @seq(%in: memref<32xi32>, %oa: memref<24xi32>, %ob: memref<8xi32>) {
      // One 8-element slice per id, sent in the order the receive buffers
      // expect: 0 -> a0, 3 -> a1, 4 -> a2, then 1 -> b0. Only the last BD issues
      // a token: the shim drains its queue in order, so waiting on that one
      // means all four have left.
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1], packet = <pkt_id = 0, pkt_type = 0>) {id = 0 : i64, metadata = @in} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 8][1, 1, 1, 8][0, 0, 0, 1], packet = <pkt_id = 3, pkt_type = 0>) {id = 1 : i64, metadata = @in} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 16][1, 1, 1, 8][0, 0, 0, 1], packet = <pkt_id = 4, pkt_type = 0>) {id = 2 : i64, metadata = @in} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 24][1, 1, 1, 8][0, 0, 0, 1], packet = <pkt_id = 1, pkt_type = 0>) {id = 3 : i64, issue_token = true, metadata = @in} : memref<32xi32>
      aiex.npu.dma_wait {symbol = @in}
      // Everything the shim was going to send has left; release the readbacks.
      aiex.set_lock(%go_a, 3)
      aiex.set_lock(%go_b, 1)
      aiex.npu.dma_memcpy_nd(%oa[0, 0, 0, 0][1, 1, 1, 24][0, 0, 0, 1]) {id = 4 : i64, issue_token = true, metadata = @outa} : memref<24xi32>
      aiex.npu.dma_memcpy_nd(%ob[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 5 : i64, issue_token = true, metadata = @outb} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @outa}
      aiex.npu.dma_wait {symbol = @outb}
    }
  }
}
