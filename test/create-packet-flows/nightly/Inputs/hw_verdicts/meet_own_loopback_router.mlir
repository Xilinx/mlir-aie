// HW: PASS
// router_hw.py meet_own_loopback, router: the router's routing without allow-deadlock-prone: id 31 meets id 30 at (0,1) by a U-turn. Observed on NPU Strix: 20 PASS.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    %t0_3 = aie.tile(0, 3)
    aie.flow(%t0_2, DMA : 0, %t0_1, DMA : 0)
    aie.packet_flow(28) { aie.packet_source<%t0_1, DMA : 3> aie.packet_dest<%t0_2, DMA : 0> }
    aie.packet_flow(30) { aie.packet_source<%t0_1, DMA : 3> aie.packet_dest<%t0_1, DMA : 3> aie.packet_dest<%t0_2, DMA : 0> }
    aie.packet_flow(31) { aie.packet_source<%t0_3, DMA : 1> aie.packet_dest<%t0_1, DMA : 3> aie.packet_dest<%t0_2, DMA : 0> }
    aie.flow(%t0_1, DMA : 2, %t0_0, DMA : 0)
    aie.flow(%t0_2, DMA : 1, %t0_0, DMA : 1)
    %src = aie.buffer(%t0_1) {sym_name = "src"} : memref<128xi32>
    %src_free = aie.lock(%t0_1, 0) {init = 2 : i32, sym_name = "src_free"}
    %src_full = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "src_full"}
    %mt = aie.buffer(%t0_1) {sym_name = "mt"} : memref<128xi32>
    %mt_free = aie.lock(%t0_1, 2) {init = 1 : i32, sym_name = "mt_free"}
    %mt_full = aie.lock(%t0_1, 3) {init = 0 : i32, sym_name = "mt_full"}
    %fill = aie.buffer(%t0_2) {sym_name = "fill"} : memref<128xi32>
    %fill_full = aie.lock(%t0_2, 0) {init = 0 : i32, sym_name = "fill_full"}
    %fill_sent = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "fill_sent"}
    %d2 = aie.buffer(%t0_2) {sym_name = "d2"} : memref<192xi32>
    %d2_free = aie.lock(%t0_2, 2) {init = 1 : i32, sym_name = "d2_free"}
    %d2_full = aie.lock(%t0_2, 3) {init = 0 : i32, sym_name = "d2_full"}
    %own = aie.buffer(%t0_3) {sym_name = "own"} : memref<64xi32>
    %own_full = aie.lock(%t0_3, 0) {init = 0 : i32, sym_name = "own_full"}
    %own_sent = aie.lock(%t0_3, 1) {init = 0 : i32, sym_name = "own_sent"}
    %core_t0_2 = aie.core(%t0_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 128 : index
      %base = arith.constant 5000 : i32
      %k = arith.constant 11 : i32
      scf.for %i = %c0 to %cn step %c1 {
        %i32 = arith.index_cast %i : index to i32
        %t = arith.muli %i32, %k : i32
        %v = arith.addi %base, %t : i32
        memref.store %v, %fill[%i] : memref<128xi32>
      }
      aie.use_lock(%fill_full, Release, %one)
      aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^s1)
      ^b0:
        aie.use_lock(%fill_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%fill : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%fill_sent, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(S2MM, 0, ^r0, ^s2)
      ^r0:
        aie.use_lock(%d2_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%d2 : memref<192xi32> offset = 0 len = 192)
        aie.use_lock(%d2_full, Release, %one)
        aie.next_bd ^end
      ^s2:
      %d2s = aie.dma_start(MM2S, 1, ^w0, ^end)
      ^w0:
        aie.use_lock(%d2_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%d2 : memref<192xi32> offset = 0 len = 192)
        aie.use_lock(%d2_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
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
        memref.store %v, %own[%i] : memref<64xi32>
      }
      aie.use_lock(%own_full, Release, %one)
      aie.end
    }
    %dma_t0_3 = aie.mem(%t0_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^b0, ^end)
      ^b0:
        aie.use_lock(%own_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%own : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
        aie.use_lock(%own_sent, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      %d0 = aie.dma_start(S2MM, 0, ^in0, ^s1)
      ^in0:
        aie.use_lock(%src_free, AcquireGreaterEqual, %two)
        aie.dma_bd(%src : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%src_full, Release, %two)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 3, ^send0, ^s2)
      ^send0:
        aie.use_lock(%src_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%src : memref<128xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 28>}
        aie.use_lock(%src_free, Release, %one)
        aie.next_bd ^send1
      ^send1:
        aie.use_lock(%src_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%src : memref<128xi32> offset = 64 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
        aie.use_lock(%src_free, Release, %one)
        aie.next_bd ^end
      ^s2:
      %dd2 = aie.dma_start(S2MM, 3, ^r0, ^s3)
      ^r0:
        aie.use_lock(%mt_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%mt_full, Release, %one)
        aie.next_bd ^end
      ^s3:
      %dd3 = aie.dma_start(MM2S, 2, ^w0, ^end)
      ^w0:
        aie.use_lock(%mt_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<128xi32> offset = 0 len = 128)
        aie.use_lock(%mt_free, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t0_0, S2MM, 1)
    aie.runtime_sequence(%o0: memref<128xi32>, %o1: memref<192xi32>) {
      aiex.npu.dma_memcpy_nd(%o0[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%o1[0, 0, 0, 0][1, 1, 1, 192][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out1} : memref<192xi32>
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
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 0, %0)
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 31, %0)
      }
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<North : 0, DMA : 0>
      aie.connect<DMA : 2, South : 2>
      aie.connect<North : 3, South : 3>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.masterset(DMA : 3, %1)
      %3 = aie.masterset(North : 1, %0, %1)
      aie.packet_rules(North : 1) {
        aie.rule(31, 31, %1)
      }
      aie.packet_rules(DMA : 3) {
        aie.rule(31, 30, %1)
        aie.rule(31, 28, %0)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, South : 0>
      aie.connect<DMA : 1, South : 3>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %0)
      %3 = aie.masterset(South : 1, %1)
      aie.packet_rules(South : 1) {
        aie.rule(28, 28, %0)
      }
      aie.packet_rules(North : 0) {
        aie.rule(31, 31, %1)
      }
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<North : 2, South : 2>
      aie.connect<North : 3, South : 3>
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
  }
}
