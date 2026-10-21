#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Two packet trees that meet to cover a port's loopback into itself.

The memtile (0,1) sends what core (0,2) made as id 18 to its own S2MM 2,
(0,3) and (0,5); core (0,4) sends its own data as id 28 to the same three.
Trees with ids of their own into receivers they share wait on each other
there, so they meet, and the meeting takes id 28 into all three, the port id
18 comes back into included. Each receiver forwards both packets to the host.
Reduced from router_properties.py npu1 seed 207461204, moved to npu2.
"""

import numpy as np

from hw_case import HwCase, Variant

CHUNK = 64
PACKETS = [
    5000 + 11 * np.arange(CHUNK, dtype=np.int32),
    1000 + 7 * np.arange(CHUNK, dtype=np.int32),
]

DESIGN = """module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t1_0 = aie.tile(1, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    %t0_3 = aie.tile(0, 3)
    %t0_4 = aie.tile(0, 4)
    %t0_5 = aie.tile(0, 5)
    aie.flow(%t0_2, DMA : 0, %t0_1, DMA : 0)
    aie.packet_flow(18) { aie.packet_source<%t0_1, DMA : 2> aie.packet_dest<%t0_1, DMA : 2> aie.packet_dest<%t0_3, DMA : 1> aie.packet_dest<%t0_5, DMA : 0> }
    aie.packet_flow(28) { aie.packet_source<%t0_4, DMA : 1> aie.packet_dest<%t0_1, DMA : 2> aie.packet_dest<%t0_3, DMA : 1> aie.packet_dest<%t0_5, DMA : 0> }
    aie.flow(%t0_1, DMA : 3, %t0_0, DMA : 0)
    aie.flow(%t0_3, DMA : 0, %t0_0, DMA : 1)
    aie.flow(%t0_5, DMA : 1, %t1_0, DMA : 0)
    %src = aie.buffer(%t0_1) {sym_name = "src"} : memref<64xi32>
    %src_free = aie.lock(%t0_1, 0) {init = 1 : i32, sym_name = "src_free"}
    %src_full = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "src_full"}
    %mt = aie.buffer(%t0_1) {sym_name = "mt"} : memref<128xi32>
    %mt_free = aie.lock(%t0_1, 2) {init = 1 : i32, sym_name = "mt_free"}
    %mt_full = aie.lock(%t0_1, 3) {init = 0 : i32, sym_name = "mt_full"}
    %own = aie.buffer(%t0_4) {sym_name = "own"} : memref<64xi32>
    %own_full = aie.lock(%t0_4, 0) {init = 0 : i32, sym_name = "own_full"}
    %own_sent = aie.lock(%t0_4, 1) {init = 0 : i32, sym_name = "own_sent"}
    %d3 = aie.buffer(%t0_3) {sym_name = "d3"} : memref<128xi32>
    %d3_free = aie.lock(%t0_3, 0) {init = 1 : i32, sym_name = "d3_free"}
    %d3_full = aie.lock(%t0_3, 1) {init = 0 : i32, sym_name = "d3_full"}
    %d5 = aie.buffer(%t0_5) {sym_name = "d5"} : memref<128xi32>
    %d5_free = aie.lock(%t0_5, 0) {init = 1 : i32, sym_name = "d5_free"}
    %d5_full = aie.lock(%t0_5, 1) {init = 0 : i32, sym_name = "d5_full"}
    %fill = aie.buffer(%t0_2) {sym_name = "fill"} : memref<64xi32>
    %fill_full = aie.lock(%t0_2, 0) {init = 0 : i32, sym_name = "fill_full"}
    %fill_sent = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "fill_sent"}
    %core_t0_2 = aie.core(%t0_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 64 : index
      %base = arith.constant 5000 : i32
      %k = arith.constant 11 : i32
      scf.for %i = %c0 to %cn step %c1 {
        %i32 = arith.index_cast %i : index to i32
        %t = arith.muli %i32, %k : i32
        %v = arith.addi %base, %t : i32
        memref.store %v, %fill[%i] : memref<64xi32>
      }
      aie.use_lock(%fill_full, Release, %one)
      aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^end)
      ^b0:
        aie.use_lock(%fill_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%fill : memref<64xi32> offset = 0 len = 64)
        aie.use_lock(%fill_sent, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %core_t0_4 = aie.core(%t0_4) {
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
    %dma_t0_4 = aie.mem(%t0_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^b0, ^end)
      ^b0:
        aie.use_lock(%own_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%own : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 28>}
        aie.use_lock(%own_sent, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^in0, ^s1)
      ^in0:
        aie.use_lock(%src_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%src : memref<64xi32> offset = 0 len = 64)
        aie.use_lock(%src_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 2, ^send, ^s2)
      ^send:
        aie.use_lock(%src_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%src : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 18>}
        aie.use_lock(%src_free, Release, %one)
        aie.next_bd ^end
      ^s2:
      %dd2 = aie.dma_start(S2MM, 2, ^r0, ^s3)
      ^r0:
        aie.use_lock(%mt_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%mt_full, Release, %one)
        aie.next_bd ^end
      ^s3:
      %dd3 = aie.dma_start(MM2S, 3, ^w0, ^end)
      ^w0:
        aie.use_lock(%mt_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%mt_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_3 = aie.mem(%t0_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 1, ^r0, ^s1)
      ^r0:
        aie.use_lock(%d3_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%d3 : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%d3_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%d3_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%d3 : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%d3_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_5 = aie.mem(%t0_5) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%d5_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%d5 : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%d5_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 1, ^w0, ^end)
      ^w0:
        aie.use_lock(%d5_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%d5 : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%d5_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t0_0, S2MM, 1)
    aie.shim_dma_allocation @out2(%t1_0, S2MM, 0)
    aie.runtime_sequence(%o0: memref<128xi32>, %o1: memref<128xi32>, %o2: memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(%o0[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%o1[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out1} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%o2[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @out2} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
      aiex.npu.dma_wait {symbol = @out2}
    }
  }
}
"""


class Case(HwCase):
    name = "meet_loopback"
    description = __doc__
    outputs = [(2 * CHUNK, np.int32)] * 3

    def variants(self) -> list[Variant]:
        return [
            Variant(
                "router",
                DESIGN,
                None,
                "PASS",
                "",
                "the router's routing without allow-deadlock-prone: the trees meet at (0,1).",
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
