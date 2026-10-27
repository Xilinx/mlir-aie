#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A packet tree that comes back through its own source switchbox.

Core (0,2) fills a buffer that its DMA sends as two packets with id 5 to the
memtile below and the core tile above, and each receiver forwards what it got
to the host. The router branches the tree at (0,2). Pinned, the tree instead
goes down to the memtile and turns back up through (0,2), as the router's
trunk rule once forced in router_properties.py seed 203651000. A packet holds
its arbiter at (0,2) until its last word leaves, so if the way back up needs
that arbiter too, the packet waits on itself.
"""

import numpy as np

from hw_case import HwCase, Variant

N = 512

DESIGN = """module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    %t0_3 = aie.tile(0, 3)
@TREE@
    aie.flow(%t0_1, DMA : 0, %t0_0, DMA : 0)
    aie.flow(%t0_3, DMA : 0, %t0_0, DMA : 1)
    aie.buffer @src(%t0_2) : memref<512xi32>
    aie.lock @src_full(%t0_2, 0) {init = 0 : i32}
    aie.lock @src_sent(%t0_2, 1) {init = 0 : i32}
    aie.buffer @mt(%t0_1) : memref<512xi32>
    aie.lock @mt_full(%t0_1, 0) {init = 0 : i32}
    aie.lock @mt_free(%t0_1, 1) {init = 2 : i32}
    aie.buffer @dst(%t0_3) : memref<512xi32>
    aie.lock @dst_full(%t0_3, 0) {init = 0 : i32}
    aie.lock @dst_free(%t0_3, 1) {init = 2 : i32}
    %core_t0_2 = aie.core(%t0_2) {
      %src = aie.get_buffer @src : memref<512xi32>
      %two = arith.constant 2 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 512 : index
      %base = arith.constant 1000 : i32
      %k = arith.constant 7 : i32
      scf.for %i = %c0 to %cn step %c1 {
        %i32 = arith.index_cast %i : index to i32
        %t = arith.muli %i32, %k : i32
        %v = arith.addi %base, %t : i32
        memref.store %v, %src[%i] : memref<512xi32>
      }
      aie.use_lock(@src_full, Release, %two)
      aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %src = aie.get_buffer @src : memref<512xi32>
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^end)
      ^b0:
        aie.use_lock(@src_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%src : memref<512xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
        aie.use_lock(@src_sent, Release, %one)
        aie.next_bd ^b1
      ^b1:
        aie.use_lock(@src_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%src : memref<512xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
        aie.use_lock(@src_sent, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %mt = aie.get_buffer @mt : memref<512xi32>
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(@mt_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(@mt_full, Release, %one)
        aie.next_bd ^r1
      ^r1:
        aie.use_lock(@mt_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(@mt_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(@mt_full, AcquireGreaterEqual, %two)
        aie.dma_bd(%mt : memref<512xi32> offset = 0 len = 512)
        aie.use_lock(@mt_free, Release, %two)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_3 = aie.mem(%t0_3) {
      %dst = aie.get_buffer @dst : memref<512xi32>
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(@dst_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%dst : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(@dst_full, Release, %one)
        aie.next_bd ^r1
      ^r1:
        aie.use_lock(@dst_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%dst : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(@dst_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(@dst_full, AcquireGreaterEqual, %two)
        aie.dma_bd(%dst : memref<512xi32> offset = 0 len = 512)
        aie.use_lock(@dst_free, Release, %two)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t0_0, S2MM, 1)
    aie.runtime_sequence(%o0: memref<512xi32>, %o1: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd(%o0[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<512xi32>
      aiex.npu.dma_memcpy_nd(%o1[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out1} : memref<512xi32>
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
    }
  }
}
"""

FLOW = (
    "    aie.packet_flow(5) { aie.packet_source<%t0_2, DMA : 0> "
    "aie.packet_dest<%t0_1, DMA : 0> aie.packet_dest<%t0_3, DMA : 0> }"
)

# Down South:0 to the memtile, back up its North:0 into (0,2) South:0, then
# North:0 to (0,3). @BACK@ is the amsel the way back up takes at (0,2).
UTURN = """    %sb0_2 = aie.switchbox(%t0_2) {
      %down = aie.amsel<1> (0)
      %back = @BACK@
      %m_down = aie.masterset(South : 0, %down)
      %m_up = aie.masterset(North : 0, %back)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 5, %down)
      }
      aie.packet_rules(South : 0) {
        aie.rule(31, 5, %back)
      }
    }
    %sb0_1 = aie.switchbox(%t0_1) {
      %turn = aie.amsel<1> (0)
      %m_dma = aie.masterset(DMA : 0, %turn)
      %m_up = aie.masterset(North : 0, %turn)
      aie.packet_rules(North : 0) {
        aie.rule(31, 5, %turn)
      }
    }
    %sb0_3 = aie.switchbox(%t0_3) {
      %in = aie.amsel<1> (0)
      %m_dma = aie.masterset(DMA : 0, %in)
      aie.packet_rules(South : 0) {
        aie.rule(31, 5, %in)
      }
    }"""

OPTIONS = "circuit-switch-hops=false allow-deadlock-prone=true"


class Case(HwCase):
    name = "uturn_self_hold"
    description = __doc__
    outputs = [(N, np.int32), (N, np.int32)]

    def variants(self) -> list[Variant]:
        design = DESIGN.replace("@TREE@", FLOW)
        apart, same = (
            DESIGN.replace("@TREE@", UTURN.replace("@BACK@", amsel))
            for amsel in ("aie.amsel<2> (0)", "aie.amsel<1> (1)")
        )
        return [
            Variant(
                "router",
                design,
                None,
                "PASS",
                OPTIONS,
                "the router's routing, which branches the tree at its source.",
            ),
            Variant(
                "uturn_apart",
                design,
                apart,
                "PASS",
                OPTIONS,
                "the tree turns back up through (0,2) on an arbiter of its own there.",
            ),
            Variant(
                "uturn_same",
                design,
                same,
                "HANG",
                OPTIONS,
                "the tree turns back up through (0,2) on the arbiter it already holds there.",
            ),
        ]

    def check(self, outputs: list[np.ndarray]) -> list[str]:
        want = 1000 + 7 * np.arange(N, dtype=np.int32)
        problems = []
        for k, got in enumerate(outputs):
            bad = np.flatnonzero(got != want)
            if bad.size:
                i = bad[0]
                problems.append(
                    f"output {k}[{i}] = {got[i]}, want {want[i]} ({bad.size} wrong)"
                )
        return problems
