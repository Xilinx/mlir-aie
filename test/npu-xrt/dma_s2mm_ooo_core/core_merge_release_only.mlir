//===- core_merge_release_only.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A compute core merges an out-of-order S2MM (two shim senders) and drains it,
// completion signalled on-chip by a release-only counting lock (each receive BD
// releases ooo_cons +1; the egress MM2S acquires n). No completion token, no
// host round-trip.
//
module {
  aie.device(npu2) {
    %s0   = aie.tile(0, 0)  // shim sender 0
    %s1   = aie.tile(2, 0)  // shim sender 1
    %core = aie.tile(0, 2)  // merge consumer core
    %egr  = aie.tile(1, 0)  // egress shim

    %inbuf  = aie.buffer(%core) {sym_name = "inbuf"}  : memref<32xi32>
    %outbuf = aie.buffer(%core) {sym_name = "outbuf"} : memref<32xi32>

    // ooo_cons: recv BDs release +1 each; core acquires >= n (release-only
    // completion counter -- no acquire on the OoO recv BDs, so no inter-BD dep).
    %ooo_cons = aie.lock(%core, 0) {init = 0 : i32, sym_name = "ooo_cons"}
    // outbuf producer/consumer between the core program and the core MM2S.
    %out_prod = aie.lock(%core, 2) {init = 1 : i32, sym_name = "out_prod"}
    %out_cons = aie.lock(%core, 3) {init = 0 : i32, sym_name = "out_cons"}

    // Ingress: 2 packet flows merge into the core's S2MM ch0 (keep header id).
    aie.packet_flow(0x0) {
      aie.packet_source<%s0, DMA : 0>
      aie.packet_dest<%core, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0x0) {
      aie.packet_source<%s1, DMA : 0>
      aie.packet_dest<%core, DMA : 0>
    } {keep_pkt_header = true}
    // Egress: core MM2S ch1 -> egress shim S2MM ch0.
    aie.flow(%core, DMA : 1, %egr, DMA : 0)

    %c = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %two = arith.constant 2 : i32
      %one = arith.constant 1 : i32
      // wait for the whole merge (2 packets placed by ooo id)
      aie.use_lock(%ooo_cons, AcquireGreaterEqual, %two)
      // copy merged inbuf -> outbuf under the outbuf producer lock
      aie.use_lock(%out_prod, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c32 step %c1 {
        %v = memref.load %inbuf[%i] : memref<32xi32>
        memref.store %v, %outbuf[%i] : memref<32xi32>
      }
      aie.use_lock(%out_cons, Release, %one)
      aie.end
    }

    %m = aie.mem(%core) {
      %0 = aie.dma_start(S2MM, 0, ^recv0, ^mm2s, repeat_count = 2) {out_of_order}
    ^recv0:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%inbuf : memref<32xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      %r0 = arith.constant 1 : i32
      aie.use_lock(%ooo_cons, Release, %r0)
      aie.next_bd ^recv1
    ^recv1:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%inbuf : memref<32xi32> offset = 16 len = 16) {bd_id = 1 : i32}
      %r1 = arith.constant 1 : i32
      aie.use_lock(%ooo_cons, Release, %r1)
      aie.next_bd ^recv0
    ^mm2s:
      %1 = aie.dma_start(MM2S, 1, ^send, ^end)
    ^send:
      %s_one = arith.constant 1 : i32
      aie.use_lock(%out_cons, AcquireGreaterEqual, %s_one)
      aie.dma_bd(%outbuf : memref<32xi32>) {bd_id = 2 : i32}
      %s_one2 = arith.constant 1 : i32
      aie.use_lock(%out_prod, Release, %s_one2)
      aie.next_bd ^send
    ^end:
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<32xi32>, %arg1: memref<32xi32>) {
      // Sender 0 (col 0): stamp out-of-order id 1, source = a_in[0:16).
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 16 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 1 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %sa0 = arith.constant 0 : i32
      aiex.npu.address_patch(%sa0 : i32) {addr = 118788 : ui32, arg_idx = 0 : i32}
      %sq0 = arith.constant 0 : i32
      %sr0 = arith.constant 0 : i32
      aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %sq0 repeat %sr0 {issue_token = true} : i32, i32
      // Sender 1 (col 2): stamp out-of-order id 0, source = a_in[16:32).
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 16 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %sa1 = arith.constant 64 : i32
      aiex.npu.address_patch(%sa1 : i32) {addr = 67227652 : ui32, arg_idx = 0 : i32}
      %sq1 = arith.constant 0 : i32
      %sr1 = arith.constant 0 : i32
      aiex.npu.push_queue(2, 0, MM2S : 0) bd_id %sq1 repeat %sr1 {issue_token = true} : i32, i32
      // Sync both senders (their MM2S completes = data injected).
      %y0c = arith.constant 0 : i32
      %y0r = arith.constant 0 : i32
      %y0d = arith.constant 1 : i32
      %y0h = arith.constant 0 : i32
      %y0a = arith.constant 1 : i32
      %y0b = arith.constant 1 : i32
      aiex.npu.sync(%y0c, %y0r, %y0d, %y0h, %y0a, %y0b) : i32, i32, i32, i32, i32, i32
      %y1c = arith.constant 2 : i32
      %y1r = arith.constant 0 : i32
      %y1d = arith.constant 1 : i32
      %y1h = arith.constant 0 : i32
      %y1a = arith.constant 1 : i32
      %y1b = arith.constant 1 : i32
      aiex.npu.sync(%y1c, %y1r, %y1d, %y1h, %y1a, %y1b) : i32, i32, i32, i32, i32, i32
      // Egress: host-driven shim S2MM (col 1) drains outbuf -> c_out.
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 32 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %ea = arith.constant 0 : i32
      aiex.npu.address_patch(%ea : i32) {addr = 33673220 : ui32, arg_idx = 1 : i32}
      %eq = arith.constant 0 : i32
      %er = arith.constant 0 : i32
      aiex.npu.push_queue(1, 0, S2MM : 0) bd_id %eq repeat %er {issue_token = true} : i32, i32
      %syc = arith.constant 1 : i32
      %syr = arith.constant 0 : i32
      %syd = arith.constant 0 : i32
      %syh = arith.constant 0 : i32
      %sya = arith.constant 1 : i32
      %syb = arith.constant 1 : i32
      aiex.npu.sync(%syc, %syr, %syd, %syh, %sya, %syb) : i32, i32, i32, i32, i32, i32
    }
  }
}
