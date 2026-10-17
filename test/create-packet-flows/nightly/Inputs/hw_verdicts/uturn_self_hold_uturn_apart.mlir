// HW: PASS
// hops: off
// router_hw.py uturn_self_hold, uturn_apart: the tree turns back up through (0,2) on an arbiter of its own there. Observed on an NPU2 (Strix): 20 PASS.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    %t0_3 = aie.tile(0, 3)
    aie.packet_flow(5) { aie.packet_source<%t0_2, DMA : 0> aie.packet_dest<%t0_1, DMA : 0> aie.packet_dest<%t0_3, DMA : 0> }
    aie.flow(%t0_1, DMA : 0, %t0_0, DMA : 0)
    aie.flow(%t0_3, DMA : 0, %t0_0, DMA : 1)
    %src = aie.buffer(%t0_2) {sym_name = "src"} : memref<512xi32>
    %src_full = aie.lock(%t0_2, 0) {init = 0 : i32, sym_name = "src_full"}
    %src_sent = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "src_sent"}
    %mt = aie.buffer(%t0_1) {sym_name = "mt"} : memref<512xi32>
    %mt_full = aie.lock(%t0_1, 0) {init = 0 : i32, sym_name = "mt_full"}
    %mt_free = aie.lock(%t0_1, 1) {init = 2 : i32, sym_name = "mt_free"}
    %dst = aie.buffer(%t0_3) {sym_name = "dst"} : memref<512xi32>
    %dst_full = aie.lock(%t0_3, 0) {init = 0 : i32, sym_name = "dst_full"}
    %dst_free = aie.lock(%t0_3, 1) {init = 2 : i32, sym_name = "dst_free"}
    %core_t0_2 = aie.core(%t0_2) {
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
      aie.use_lock(%src_full, Release, %two)
      aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^end)
      ^b0:
        aie.use_lock(%src_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%src : memref<512xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
        aie.use_lock(%src_sent, Release, %one)
        aie.next_bd ^b1
      ^b1:
        aie.use_lock(%src_full, AcquireGreaterEqual, %one)
        aie.dma_bd(%src : memref<512xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
        aie.use_lock(%src_sent, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%mt_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%mt_full, Release, %one)
        aie.next_bd ^r1
      ^r1:
        aie.use_lock(%mt_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%mt_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%mt_full, AcquireGreaterEqual, %two)
        aie.dma_bd(%mt : memref<512xi32> offset = 0 len = 512)
        aie.use_lock(%mt_free, Release, %two)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_3 = aie.mem(%t0_3) {
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      %d0 = aie.dma_start(S2MM, 0, ^r0, ^s1)
      ^r0:
        aie.use_lock(%dst_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%dst : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%dst_full, Release, %one)
        aie.next_bd ^r1
      ^r1:
        aie.use_lock(%dst_free, AcquireGreaterEqual, %one)
        aie.dma_bd(%dst : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%dst_full, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^w0, ^end)
      ^w0:
        aie.use_lock(%dst_full, AcquireGreaterEqual, %two)
        aie.dma_bd(%dst : memref<512xi32> offset = 0 len = 512)
        aie.use_lock(%dst_free, Release, %two)
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
// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<1> (0)
      %1 = aie.amsel<2> (0)
      %2 = aie.masterset(South : 0, %0)
      %3 = aie.masterset(North : 0, %1)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 5, %0)
      }
      aie.packet_rules(South : 0) {
        aie.rule(31, 5, %1)
      }
      aie.connect<North : 1, South : 2>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<1> (0)
      %1 = aie.masterset(DMA : 0, %0)
      %2 = aie.masterset(North : 0, %0)
      aie.packet_rules(North : 0) {
        aie.rule(31, 5, %0)
      }
      aie.connect<DMA : 0, South : 3>
      aie.connect<North : 2, South : 2>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<1> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(South : 0) {
        aie.rule(31, 5, %0)
      }
      aie.connect<DMA : 0, South : 1>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<North : 3, South : 2>
      aie.connect<North : 2, South : 3>
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
  }
}
