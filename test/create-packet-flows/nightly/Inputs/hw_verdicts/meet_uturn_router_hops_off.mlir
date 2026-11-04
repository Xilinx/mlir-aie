// HW: PASS
// hops: off
// router_hw.py meet_uturn, router_hops_off: the same, with no packet hop circuit switched. Observed on NPU Strix: 20 PASS.
module {
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
// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<North : 2, South : 2>
      aie.connect<North : 3, South : 3>
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 0, South : 2>
      aie.connect<North : 3, South : 3>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(North : 0) {
        aie.rule(31, 28, %0)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<North : 2, South : 3>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %1)
      %3 = aie.masterset(South : 0, %0)
      %4 = aie.masterset(North : 3, %0)
      aie.packet_rules(North : 3) {
        aie.rule(31, 14, %1)
        aie.rule(31, 28, %0)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 28, %0)
      }
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<North : 0, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 3, %0)
      %3 = aie.masterset(North : 0, %1)
      aie.packet_rules(DMA : 0) {
        aie.rule(13, 12, %0)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 28, %1)
      }
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(South : 0) {
        aie.rule(31, 28, %0)
      }
    }
  }
}
