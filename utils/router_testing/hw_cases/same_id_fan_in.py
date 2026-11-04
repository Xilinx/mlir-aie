#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Two packet flows with one id that fan in only at the receiver they share.

Core (0,3) sends id 24 to the memtile (2,1) and to (2,3); core (2,4) sends
two packets with id 24, one from each DMA channel, to (0,4), the memtile and
(2,2). The trees come down column 2 on channels of their own and meet only
at the memtile's S2MM 0. The receivers forward the packets they get to the
host.
Reduced from router_mutation.py's id-reuse mutation of router_properties.py
npu1 seed 207460302, moved to npu2.
"""

import numpy as np

from hw_case import HwCase, Variant

CHUNK = 64
A = 1000 + 7 * np.arange(CHUNK, dtype=np.int32)
C = 5000 + 11 * np.arange(CHUNK, dtype=np.int32)
D = 7000 + 13 * np.arange(CHUNK, dtype=np.int32)

DESIGN = """module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t2_0 = aie.tile(2, 0)
    %t3_0 = aie.tile(3, 0)
    %t0_3 = aie.tile(0, 3)
    %t0_4 = aie.tile(0, 4)
    %t2_1 = aie.tile(2, 1)
    %t2_2 = aie.tile(2, 2)
    %t2_3 = aie.tile(2, 3)
    %t2_4 = aie.tile(2, 4)
    aie.packet_flow(24) { aie.packet_source<%t0_3, DMA : 0> aie.packet_dest<%t2_1, DMA : 0> aie.packet_dest<%t2_3, DMA : 0> }
    aie.packet_flow(24) { aie.packet_source<%t2_4, DMA : 0> aie.packet_source<%t2_4, DMA : 1> aie.packet_dest<%t0_4, DMA : 0> aie.packet_dest<%t2_1, DMA : 0> aie.packet_dest<%t2_2, DMA : 0> }
    aie.flow(%t2_1, DMA : 0, %t2_0, DMA : 0)
    aie.flow(%t2_3, DMA : 0, %t2_0, DMA : 1)
    aie.flow(%t0_4, DMA : 0, %t0_0, DMA : 0)
    aie.flow(%t2_2, DMA : 0, %t3_0, DMA : 0)
    %a = aie.buffer(%t0_3) {sym_name = "a"} : memref<64xi32>
    %a_full = aie.lock(%t0_3, 0) {init = 0 : i32, sym_name = "a_full"}
    %a_sent = aie.lock(%t0_3, 1) {init = 0 : i32, sym_name = "a_sent"}
    %c = aie.buffer(%t2_4) {sym_name = "c"} : memref<64xi32>
    %c_full = aie.lock(%t2_4, 0) {init = 0 : i32, sym_name = "c_full"}
    %c_sent = aie.lock(%t2_4, 1) {init = 0 : i32, sym_name = "c_sent"}
    %d = aie.buffer(%t2_4) {sym_name = "d"} : memref<64xi32>
    %d_full = aie.lock(%t2_4, 2) {init = 0 : i32, sym_name = "d_full"}
    %d_sent = aie.lock(%t2_4, 3) {init = 0 : i32, sym_name = "d_sent"}
    %r1 = aie.buffer(%t2_1) {sym_name = "r1"} : memref<192xi32>
    %r1_free = aie.lock(%t2_1, 0) {init = 1 : i32, sym_name = "r1_free"}
    %r1_full = aie.lock(%t2_1, 1) {init = 0 : i32, sym_name = "r1_full"}
    %r2 = aie.buffer(%t2_2) {sym_name = "r2"} : memref<128xi32>
    %r2_free = aie.lock(%t2_2, 0) {init = 1 : i32, sym_name = "r2_free"}
    %r2_full = aie.lock(%t2_2, 1) {init = 0 : i32, sym_name = "r2_full"}
    %r3 = aie.buffer(%t2_3) {sym_name = "r3"} : memref<64xi32>
    %r3_free = aie.lock(%t2_3, 0) {init = 1 : i32, sym_name = "r3_free"}
    %r3_full = aie.lock(%t2_3, 1) {init = 0 : i32, sym_name = "r3_full"}
    %r4 = aie.buffer(%t0_4) {sym_name = "r4"} : memref<128xi32>
    %r4_free = aie.lock(%t0_4, 0) {init = 1 : i32, sym_name = "r4_free"}
    %r4_full = aie.lock(%t0_4, 1) {init = 0 : i32, sym_name = "r4_full"}
    %core_t0_3 = aie.core(%t0_3) {
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
        memref.store %v, %a[%i] : memref<64xi32>
      }
      aie.use_lock(%a_full, Release, %one)
      aie.end
    }
    %dma_t0_3 = aie.mem(%t0_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^end)
      ^b0:
        aie.use_lock(%a_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%a : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 24>}
        aie.use_lock(%a_sent, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %core_t2_4 = aie.core(%t2_4) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 64 : index
      %base_c = arith.constant 5000 : i32
      %k_c = arith.constant 11 : i32
      %base_d = arith.constant 7000 : i32
      %k_d = arith.constant 13 : i32
      scf.for %i = %c0 to %cn step %c1 {
        %i32 = arith.index_cast %i : index to i32
        %tc = arith.muli %i32, %k_c : i32
        %vc = arith.addi %base_c, %tc : i32
        memref.store %vc, %c[%i] : memref<64xi32>
        %td = arith.muli %i32, %k_d : i32
        %vd = arith.addi %base_d, %td : i32
        memref.store %vd, %d[%i] : memref<64xi32>
      }
      aie.use_lock(%c_full, Release, %one)
      aie.use_lock(%d_full, Release, %one)
      aie.end
    }
    %dma_t2_4 = aie.mem(%t2_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^s1)
      ^b0:
        aie.use_lock(%c_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%c : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 24>}
        aie.use_lock(%c_sent, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 1, ^b1, ^end)
      ^b1:
        aie.use_lock(%d_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%d : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 24>}
        aie.use_lock(%d_sent, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t2_1 = aie.memtile_dma(%t2_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%r1_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%r1 : memref<192xi32> offset = 0 len = 192)
        aie.use_lock(%r1_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%r1_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%r1 : memref<192xi32> offset = 0 len = 192)
        aie.use_lock(%r1_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t2_2 = aie.mem(%t2_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%r2_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%r2 : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%r2_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%r2_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%r2 : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%r2_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t2_3 = aie.mem(%t2_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%r3_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%r3 : memref<64xi32> offset = 0 len = 64)
        aie.use_lock(%r3_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%r3_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%r3 : memref<64xi32> offset = 0 len = 64)
        aie.use_lock(%r3_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_4 = aie.mem(%t0_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%r4_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%r4 : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%r4_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%r4_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%r4 : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%r4_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @out0(%t2_0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t2_0, S2MM, 1)
    aie.shim_dma_allocation @out2(%t0_0, S2MM, 0)
    aie.shim_dma_allocation @out3(%t3_0, S2MM, 0)
    aie.runtime_sequence(%o0: memref<192xi32>, %o1: memref<64xi32>, %o2: memref<128xi32>, %o3: memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(%o0[0, 0, 0, 0][1, 1, 1, 192][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<192xi32>
      aiex.npu.dma_memcpy_nd(%o1[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out1} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%o2[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @out2} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%o3[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 3 : i64, issue_token = true, metadata = @out3} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
      aiex.npu.dma_wait {symbol = @out2}
      aiex.npu.dma_wait {symbol = @out3}
    }
  }
}
"""


class Case(HwCase):
    name = "same_id_fan_in"
    description = __doc__
    outputs = [
        (3 * CHUNK, np.int32),
        (CHUNK, np.int32),
        (2 * CHUNK, np.int32),
        (2 * CHUNK, np.int32),
    ]

    def variants(self) -> list[Variant]:
        return [
            Variant(
                "router",
                DESIGN,
                None,
                "PASS",
                "",
                "the router's routing without allow-deadlock-prone: the trees fan in at (2,1) alone.",
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
        return self.check_packets(outputs, [[A, C, D], [A], [C, D], [C, D]], CHUNK)
