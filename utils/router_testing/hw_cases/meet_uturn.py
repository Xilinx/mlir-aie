#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Two packet trees that meet through a same-channel U-turn.

Core (0,3) sends id 14 to (0,2) and two packets with id 28 to the memtile
(0,1) and to (0,4); (0,2) relays what id 14 brought, then its own data, as
id 28 to the same two. The trees meet at (0,2): (0,3)'s id 28 comes down
into (0,2) and leaves by its master ports, up again by the North channel it
came in on. The memtile and (0,4) forward the four packets they get to the
host.
"""

import numpy as np

from hw_case import HwCase, Variant

CHUNK = 64
B = 2000 + 3 * np.arange(3 * CHUNK, dtype=np.int32)
PACKETS = [
    B[:CHUNK],
    B[CHUNK : 2 * CHUNK],
    B[2 * CHUNK :],
    1000 + 7 * np.arange(CHUNK, dtype=np.int32),
]

DESIGN = """module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    %t0_3 = aie.tile(0, 3)
    %t0_4 = aie.tile(0, 4)
    aie.packet_flow(28) { aie.packet_source<%t0_2, DMA : 0> aie.packet_source<%t0_3, DMA : 0> aie.packet_dest<%t0_1, DMA : 0> aie.packet_dest<%t0_4, DMA : 0> }
    aie.packet_flow(14) { aie.packet_source<%t0_3, DMA : 0> aie.packet_dest<%t0_2, DMA : 0> }
    aie.flow(%t0_1, DMA : 0, %t0_0, DMA : 0)
    aie.flow(%t0_4, DMA : 0, %t0_0, DMA : 1)
    %bufB = aie.buffer(%t0_3) {sym_name = "bufB"} : memref<192xi32>
    %b_full = aie.lock(%t0_3, 0) {init = 0 : i32, sym_name = "b_full"}
    %b_sent = aie.lock(%t0_3, 1) {init = 0 : i32, sym_name = "b_sent"}
    %own = aie.buffer(%t0_2) {sym_name = "own"} : memref<64xi32>
    %own_full = aie.lock(%t0_2, 0) {init = 0 : i32, sym_name = "own_full"}
    %own_sent = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "own_sent"}
    %relay = aie.buffer(%t0_2) {sym_name = "relay"} : memref<64xi32>
    %relay_free = aie.lock(%t0_2, 2) {init = 1 : i32, sym_name = "relay_free"}
    %relay_full = aie.lock(%t0_2, 3) {init = 0 : i32, sym_name = "relay_full"}
    %mt = aie.buffer(%t0_1) {sym_name = "mt"} : memref<256xi32>
    %mt_free = aie.lock(%t0_1, 0) {init = 1 : i32, sym_name = "mt_free"}
    %mt_full = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "mt_full"}
    %dst = aie.buffer(%t0_4) {sym_name = "dst"} : memref<256xi32>
    %dst_free = aie.lock(%t0_4, 0) {init = 1 : i32, sym_name = "dst_free"}
    %dst_full = aie.lock(%t0_4, 1) {init = 0 : i32, sym_name = "dst_full"}
    %core_t0_3 = aie.core(%t0_3) {
      %three = arith.constant 3 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 192 : index
      %base = arith.constant 2000 : i32
      %k = arith.constant 3 : i32
      scf.for %i = %c0 to %cn step %c1 {
        %i32 = arith.index_cast %i : index to i32
        %t = arith.muli %i32, %k : i32
        %v = arith.addi %base, %t : i32
        memref.store %v, %bufB[%i] : memref<192xi32>
      }
      aie.use_lock(%b_full, Release, %three)
      aie.end
    }
    %core_t0_2 = aie.core(%t0_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 64 : index
      %base = arith.constant 1000 : i32
      %k = arith.constant 7 : i32
      scf.for %i = %c0 to %cn step %c1 {
        %i32 = arith.index_cast %i : index to i32
        %t = arith.muli %i32, %k : i32
        %v = arith.addi %base, %t : i32
        memref.store %v, %own[%i] : memref<64xi32>
      }
      aie.use_lock(%own_full, Release, %one)
      aie.end
    }
    %dma_t0_3 = aie.mem(%t0_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^end)
      ^b0:
        aie.use_lock(%b_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%bufB : memref<192xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 14>}
        aie.use_lock(%b_sent, Release, %one)
        aie.next_bd ^b1
      ^b1:
        aie.use_lock(%b_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%bufB : memref<192xi32> offset = 64 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 28>}
        aie.use_lock(%b_sent, Release, %one)
        aie.next_bd ^b2
      ^b2:
        aie.use_lock(%b_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%bufB : memref<192xi32> offset = 128 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 28>}
        aie.use_lock(%b_sent, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%relay_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%relay : memref<64xi32> offset = 0 len = 64)
        aie.use_lock(%relay_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%relay_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%relay : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 28>}
        aie.use_lock(%relay_free, Release, %one)
        aie.next_bd ^w1
      ^w1:
        aie.use_lock(%own_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%own : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 28>}
        aie.use_lock(%own_sent, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%mt_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%mt_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%mt_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%mt_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_4 = aie.mem(%t0_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%dst_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%dst : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%dst_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%dst_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%dst : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%dst_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t0_0, S2MM, 1)
    aie.runtime_sequence(%o0: memref<256xi32>, %o1: memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(%o0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<256xi32>
      aiex.npu.dma_memcpy_nd(%o1[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out1} : memref<256xi32>
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
    }
  }
}
"""


class Case(HwCase):
    name = "meet_uturn"
    description = __doc__
    outputs = [(4 * CHUNK, np.int32)] * 2

    def variants(self) -> list[Variant]:
        return [
            Variant(
                "router",
                DESIGN,
                None,
                "PASS",
                "",
                "the router's routing without allow-deadlock-prone: the trees meet at (0,2).",
            ),
            Variant(
                "router_hops_off",
                DESIGN,
                None,
                "PASS",
                "circuit-switch-hops=false",
                "the same, with no packet hop circuit switched.",
            ),
        ]

    def check(self, outputs: list[np.ndarray]) -> list[str]:
        # Packets from different sources reach a receiver in either order, so
        # each output must hold every expected packet once, in any order.
        want = sorted(tuple(p) for p in PACKETS)
        problems = []
        for k, got in enumerate(outputs):
            chunks = sorted(tuple(c) for c in got.reshape(-1, CHUNK))
            if chunks != want:
                missing = len(set(want) - set(chunks))
                problems.append(f"output {k}: {missing} expected packets missing")
        return problems
