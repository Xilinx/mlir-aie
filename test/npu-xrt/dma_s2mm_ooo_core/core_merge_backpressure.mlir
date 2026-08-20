//===- core_merge_backpressure.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Same core out-of-order merge, run over two generations into one reused buffer
// with a throttled (delay-loop) consumer and GUARANTEED backpressure: each
// receive BD acquires an ooo_prod credit before writing and releases ooo_cons
// after; the core acquires ooo_cons(n), consumes, then releases ooo_prod(n).
// The receive side provably waits for the consumer, so a slow consumer never
// causes an overrun.
//
module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %egr  = aie.tile(1, 0)
    %inbuf   = aie.buffer(%core) {sym_name = "inbuf"}   : memref<32xi32>
    %outbuf  = aie.buffer(%core) {sym_name = "outbuf"}  : memref<32xi32>
    %scratch = aie.buffer(%core) {sym_name = "scratch"} : memref<1xi32>
    %ooo_cons = aie.lock(%core, 0) {init = 0 : i32, sym_name = "ooo_cons"}
    %ooo_prod = aie.lock(%core, 1) {init = 2 : i32, sym_name = "ooo_prod"}
    %out_prod = aie.lock(%core, 2) {init = 1 : i32, sym_name = "out_prod"}
    %out_cons = aie.lock(%core, 3) {init = 0 : i32, sym_name = "out_cons"}
    aie.packet_flow(0x0) { aie.packet_source<%s0, DMA : 0> aie.packet_dest<%core, DMA : 0> } {keep_pkt_header = true}
    aie.flow(%core, DMA : 1, %egr, DMA : 0)
    %c = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cW = arith.constant 32 : index
      %cM = arith.constant 2 : index
      %cD = arith.constant 200000 : index
      %n_i = arith.constant 2 : i32
      %one = arith.constant 1 : i32
      %z = arith.constant 0 : i32
      scf.for %g = %c0 to %cM step %c1 {
        aie.use_lock(%ooo_cons, AcquireGreaterEqual, %n_i)
        %acc = scf.for %d = %c0 to %cD step %c1 iter_args(%a = %z) -> (i32) {
          %a2 = arith.addi %a, %one : i32
          scf.yield %a2 : i32
        }
        memref.store %acc, %scratch[%c0] : memref<1xi32>
        aie.use_lock(%out_prod, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %cW step %c1 {
          %v = memref.load %inbuf[%i] : memref<32xi32>
          memref.store %v, %outbuf[%i] : memref<32xi32>
        }
        aie.use_lock(%out_cons, Release, %one)
        aie.use_lock(%ooo_prod, Release, %n_i)
      }
      aie.end
    }
    %mem = aie.mem(%core) {
      %0 = aie.dma_start(S2MM, 0, ^recv0, ^mm2s, repeat_count = 4) {out_of_order}
    ^recv0:
      %bf0 = arith.constant 1 : i32
      aie.use_lock(%ooo_prod, AcquireGreaterEqual, %bf0)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%inbuf : memref<32xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      %dr0 = arith.constant 1 : i32
      aie.use_lock(%ooo_cons, Release, %dr0)
      aie.next_bd ^recv1
    ^recv1:
      %bf1 = arith.constant 1 : i32
      aie.use_lock(%ooo_prod, AcquireGreaterEqual, %bf1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%inbuf : memref<32xi32> offset = 16 len = 16) {bd_id = 1 : i32}
      %dr1 = arith.constant 1 : i32
      aie.use_lock(%ooo_cons, Release, %dr1)
      aie.next_bd ^recv0
    ^mm2s:
      %1 = aie.dma_start(MM2S, 1, ^send, ^end)
    ^send:
      %so = arith.constant 1 : i32
      aie.use_lock(%out_cons, AcquireGreaterEqual, %so)
      aie.dma_bd(%outbuf : memref<32xi32>) {bd_id = 2 : i32}
      %so2 = arith.constant 1 : i32
      aie.use_lock(%out_prod, Release, %so2)
      aie.next_bd ^send
    ^end:
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 16 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 1 : i32, out_of_order_id = 1 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 1 : i32, valid_bd = 1 : i32}
      %k0 = arith.constant 0 : i32
      aiex.npu.address_patch(%k0 : i32) {addr = 118788 : ui32, arg_idx = 0 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 16 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 2 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 1 : i32, valid_bd = 1 : i32}
      %k1 = arith.constant 64 : i32
      aiex.npu.address_patch(%k1 : i32) {addr = 118820 : ui32, arg_idx = 0 : i32}
      aiex.npu.writebd {bd_id = 2 : i32, buffer_length = 16 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 3 : i32, out_of_order_id = 1 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 1 : i32, valid_bd = 1 : i32}
      %k2 = arith.constant 128 : i32
      aiex.npu.address_patch(%k2 : i32) {addr = 118852 : ui32, arg_idx = 0 : i32}
      aiex.npu.writebd {bd_id = 3 : i32, buffer_length = 16 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %k3 = arith.constant 192 : i32
      aiex.npu.address_patch(%k3 : i32) {addr = 118884 : ui32, arg_idx = 0 : i32}
      %k4 = arith.constant 0 : i32
      %k5 = arith.constant 0 : i32
      aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %k4 repeat %k5 {issue_token = true} : i32, i32
      %k6 = arith.constant 0 : i32
      %k7 = arith.constant 0 : i32
      %k8 = arith.constant 1 : i32
      %k9 = arith.constant 0 : i32
      %k10 = arith.constant 1 : i32
      %k11 = arith.constant 1 : i32
      aiex.npu.sync(%k6, %k7, %k8, %k9, %k10, %k11) : i32, i32, i32, i32, i32, i32
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 64 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %k12 = arith.constant 0 : i32
      aiex.npu.address_patch(%k12 : i32) {addr = 33673220 : ui32, arg_idx = 1 : i32}
      %k13 = arith.constant 0 : i32
      %k14 = arith.constant 0 : i32
      aiex.npu.push_queue(1, 0, S2MM : 0) bd_id %k13 repeat %k14 {issue_token = true} : i32, i32
      %k15 = arith.constant 1 : i32
      %k16 = arith.constant 0 : i32
      %k17 = arith.constant 0 : i32
      %k18 = arith.constant 0 : i32
      %k19 = arith.constant 1 : i32
      %k20 = arith.constant 1 : i32
      aiex.npu.sync(%k15, %k16, %k17, %k18, %k19, %k20) : i32, i32, i32, i32, i32, i32
    }
  }
}
