//===- priority_route_task_complete_token.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN

// The id 15 flows are the routes the column control overlay adds for each
// shim's task-complete tokens; the host learns a dma_wait is done from them.
// Prioritized id 0 detours through both shims, which would put it on the
// tokens' arbiter. At (0, 0) that is a hold cycle through the host: id 0 can
// fill (1, 1) DMA:0, draining which needs id 2, which carries @in2, which the
// host issues only after the token for @in1 arrives through (0, 0). So the
// token takes another arbiter there. At (1, 0) the host waits only for @out,
// last, so sharing is safe. Alone, id 0 and the token share an arbiter at
// (0, 0), as in @ctrl_pkt_overlay, so a reload would not keep this routing.

// WARN: warning: the prioritized flows (the control overlay) take other packet rules at (0, 0) North:2 than they take alone

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0)
// CHECK-DAG:     %[[TCT:.*]] = aie.amsel<4> (3)
// CHECK-DAG:     %[[PRIO:.*]] = aie.amsel<5> (3)
// CHECK-DAG:     aie.masterset(South : 0, %[[TCT]])
// CHECK-DAG:     aie.masterset(East : 0, %[[PRIO]])
// CHECK-LABEL: aie.switchbox(%shim_noc_tile_1_0)
// CHECK-DAG:     %[[TCT1:.*]] = aie.amsel<5> (3)
// CHECK-DAG:     %[[PRIO1:.*]] = aie.amsel<5> (2)
// CHECK-DAG:     aie.masterset(South : 0, %[[TCT1]])
// CHECK-DAG:     aie.masterset(North : 3, %[[PRIO1]])

module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t1_0 = aie.tile(1, 0)
    %t0_1 = aie.tile(0, 1)
    %t1_1 = aie.tile(1, 1)
    aie.flow(%t0_0, DMA : 0, %t0_1, DMA : 0)
    aie.packet_flow(0) { aie.packet_source<%t0_1, DMA : 0> aie.packet_dest<%t1_1, DMA : 0> } {priority_route = true}
    aie.flow(%t0_0, DMA : 1, %t0_1, DMA : 1)
    aie.packet_flow(1) { aie.packet_source<%t0_1, DMA : 1> aie.packet_dest<%t1_1, DMA : 1> }
    aie.flow(%t1_0, DMA : 0, %t0_1, DMA : 2)
    aie.packet_flow(2) { aie.packet_source<%t0_1, DMA : 2> aie.packet_dest<%t1_1, DMA : 2> }
    aie.flow(%t1_1, DMA : 0, %t1_0, DMA : 0)
    %mt0 = aie.buffer(%t0_1) {sym_name = "mt0"} : memref<1024xi32>
    %mt0_p = aie.lock(%t0_1, 0) {init = 4 : i32, sym_name = "mt0_p"}
    %mt0_c = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "mt0_c"}
    %mt1 = aie.buffer(%t0_1) {sym_name = "mt1"} : memref<1024xi32>
    %mt1_p = aie.lock(%t0_1, 2) {init = 4 : i32, sym_name = "mt1_p"}
    %mt1_c = aie.lock(%t0_1, 3) {init = 0 : i32, sym_name = "mt1_c"}
    %mt2 = aie.buffer(%t0_1) {sym_name = "mt2"} : memref<1024xi32>
    %mt2_p = aie.lock(%t0_1, 4) {init = 4 : i32, sym_name = "mt2_p"}
    %mt2_c = aie.lock(%t0_1, 5) {init = 0 : i32, sym_name = "mt2_c"}
    %i11_0 = aie.buffer(%t1_1) {sym_name = "i11_0"} : memref<256xi32>
    %i11_0_p = aie.lock(%t1_1, 0) {init = 1 : i32, sym_name = "i11_0_p"}
    %i11_0_c = aie.lock(%t1_1, 1) {init = 0 : i32, sym_name = "i11_0_c"}
    %i11_1 = aie.buffer(%t1_1) {sym_name = "i11_1"} : memref<256xi32>
    %i11_1_p = aie.lock(%t1_1, 2) {init = 1 : i32, sym_name = "i11_1_p"}
    %i11_1_c = aie.lock(%t1_1, 3) {init = 0 : i32, sym_name = "i11_1_c"}
    %i11_2 = aie.buffer(%t1_1) {sym_name = "i11_2"} : memref<256xi32>
    %i11_2_p = aie.lock(%t1_1, 4) {init = 1 : i32, sym_name = "i11_2_p"}
    %i11_2_c = aie.lock(%t1_1, 5) {init = 0 : i32, sym_name = "i11_2_c"}
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^c0b2
      ^c0b2:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^c0b3
      ^c0b3:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^c1b1
      ^c1b1:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^c1b2
      ^c1b2:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^c1b3
      ^c1b3:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^c1b0
      ^s2:
      %d2 = aie.dma_start(S2MM, 1, ^c2b0, ^s3)
      ^c2b0:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^c2b1
      ^c2b1:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^c2b2
      ^c2b2:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^c2b3
      ^c2b3:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^c2b0
      ^s3:
      %d3 = aie.dma_start(MM2S, 1, ^c3b0, ^s4)
      ^c3b0:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^c3b1
      ^c3b1:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^c3b2
      ^c3b2:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^c3b3
      ^c3b3:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^c3b0
      ^s4:
      %d4 = aie.dma_start(S2MM, 2, ^c4b0, ^s5)
      ^c4b0:
        aie.use_lock(%mt2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(%mt2_c, Release, %one)
        aie.next_bd ^c4b1
      ^c4b1:
        aie.use_lock(%mt2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(%mt2_c, Release, %one)
        aie.next_bd ^c4b2
      ^c4b2:
        aie.use_lock(%mt2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(%mt2_c, Release, %one)
        aie.next_bd ^c4b3
      ^c4b3:
        aie.use_lock(%mt2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(%mt2_c, Release, %one)
        aie.next_bd ^c4b0
      ^s5:
      %d5 = aie.dma_start(MM2S, 2, ^c5b0, ^end)
      ^c5b0:
        aie.use_lock(%mt2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%mt2_p, Release, %one)
        aie.next_bd ^c5b1
      ^c5b1:
        aie.use_lock(%mt2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%mt2_p, Release, %one)
        aie.next_bd ^c5b2
      ^c5b2:
        aie.use_lock(%mt2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%mt2_p, Release, %one)
        aie.next_bd ^c5b3
      ^c5b3:
        aie.use_lock(%mt2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%mt2_p, Release, %one)
        aie.next_bd ^c5b0
      ^end:
        aie.end
    }
    %dma_t1_1 = aie.memtile_dma(%t1_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%i11_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_0_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(S2MM, 1, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%i11_1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_1_c, Release, %one)
        aie.next_bd ^c1b0
      ^s2:
      %d2 = aie.dma_start(S2MM, 2, ^c2b0, ^s3)
      ^c2b0:
        aie.use_lock(%i11_2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_2 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_2_c, Release, %one)
        aie.next_bd ^c2b0
      ^s3:
      %d3 = aie.dma_start(MM2S, 0, ^c3b0, ^end)
      ^c3b0:
        aie.use_lock(%i11_0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_0_p, Release, %one)
        aie.next_bd ^c3b1
      ^c3b1:
        aie.use_lock(%i11_1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_1_p, Release, %one)
        aie.next_bd ^c3b2
      ^c3b2:
        aie.use_lock(%i11_2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_2 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_2_p, Release, %one)
        aie.next_bd ^c3b0
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @in0(%t0_0, MM2S, 0)
    aie.shim_dma_allocation @in1(%t0_0, MM2S, 1)
    aie.shim_dma_allocation @in2(%t1_0, MM2S, 0)
    aie.shim_dma_allocation @out(%t1_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<3072xi32>, %out: memref<3072xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @in0} : memref<3072xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 3072][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out} : memref<3072xi32>
      aiex.npu.dma_wait {symbol = @in0}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 1024][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in1} : memref<3072xi32>
      aiex.npu.dma_wait {symbol = @in1}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 2048][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in2} : memref<3072xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
    aie.packet_flow(15) {
      aie.packet_source<%t0_0, TileControl : 0>
      aie.packet_dest<%t0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%t1_0, TileControl : 0>
      aie.packet_dest<%t1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
