// HW: PASS
// hops: off
// g and h share (2,0); g's receiver never stalls.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t1_0 = aie.tile(1, 0)
    %t2_0 = aie.tile(2, 0)
    %t3_0 = aie.tile(3, 0)
    %t0_1 = aie.tile(0, 1)
    %t1_2 = aie.tile(1, 2)
    aie.packet_flow(0) { aie.packet_source<%t1_0, DMA : 0> aie.packet_dest<%t1_2, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t2_0, DMA : 0> aie.packet_dest<%t1_2, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t3_0, DMA : 0> aie.packet_dest<%t0_1, DMA : 0> }
    aie.flow(%t1_2, DMA : 0, %t1_0, DMA : 0)
    aie.flow(%t0_1, DMA : 0, %t0_0, DMA : 0)
    %i12_0 = aie.buffer(%t1_2) {sym_name = "i12_0"} : memref<256xi32>
    %i12_0_p = aie.lock(%t1_2, 0) {init = 1 : i32, sym_name = "i12_0_p"}
    %i12_0_c = aie.lock(%t1_2, 1) {init = 0 : i32, sym_name = "i12_0_c"}
    %i12_1 = aie.buffer(%t1_2) {sym_name = "i12_1"} : memref<256xi32>
    %i12_1_p = aie.lock(%t1_2, 2) {init = 1 : i32, sym_name = "i12_1_p"}
    %i12_1_c = aie.lock(%t1_2, 3) {init = 0 : i32, sym_name = "i12_1_c"}
    %o12_0 = aie.buffer(%t1_2) {sym_name = "o12_0"} : memref<256xi32>
    %o12_0_p = aie.lock(%t1_2, 4) {init = 1 : i32, sym_name = "o12_0_p"}
    %o12_0_c = aie.lock(%t1_2, 5) {init = 0 : i32, sym_name = "o12_0_c"}
    %mtg = aie.buffer(%t0_1) {sym_name = "mtg"} : memref<512xi32>
    %mtg_p = aie.lock(%t0_1, 0) {init = 2 : i32, sym_name = "mtg_p"}
    %mtg_c = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "mtg_c"}
    %core_t1_2 = aie.core(%t1_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cb = arith.constant 256 : index
      %k = arith.constant 0 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(%i12_1_c, AcquireGreaterEqual, %one)
        aie.use_lock(%i12_0_c, AcquireGreaterEqual, %one)
        aie.use_lock(%o12_0_p, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %cb step %c1 {
          %a0 = arith.addi %i, %c0 : index
          %x0 = memref.load %i12_1[%a0] : memref<256xi32>
          %s0 = arith.addi %k, %x0 : i32
          %a1 = arith.addi %i, %c0 : index
          %x1 = memref.load %i12_0[%a1] : memref<256xi32>
          %s1 = arith.addi %s0, %x1 : i32
          %ao = arith.addi %i, %c0 : index
          memref.store %s1, %o12_0[%ao] : memref<256xi32>
        }
        aie.use_lock(%i12_1_p, Release, %one)
        aie.use_lock(%i12_0_p, Release, %one)
        aie.use_lock(%o12_0_c, Release, %one)
      }
      aie.end
    }
    %dma_t1_2 = aie.mem(%t1_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1, repeat_count = 3)
      ^c0b0:
        aie.use_lock(%i12_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i12_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i12_0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(S2MM, 1, ^c1b0, ^s2, repeat_count = 3)
      ^c1b0:
        aie.use_lock(%i12_1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i12_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i12_1_c, Release, %one)
        aie.next_bd ^end
      ^s2:
      %d2 = aie.dma_start(MM2S, 0, ^c2b0, ^end, repeat_count = 3)
      ^c2b0:
        aie.use_lock(%o12_0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%o12_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%o12_0_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1, repeat_count = 1)
      ^c0b0:
        aie.use_lock(%mtg_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mtg : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%mtg_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%mtg_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mtg : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%mtg_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end, repeat_count = 1)
      ^c1b0:
        aie.use_lock(%mtg_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mtg : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%mtg_p, Release, %one)
        aie.next_bd ^c1b1
      ^c1b1:
        aie.use_lock(%mtg_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mtg : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%mtg_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @inf(%t1_0, MM2S, 0)
    aie.shim_dma_allocation @ing(%t3_0, MM2S, 0)
    aie.shim_dma_allocation @inh(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @outc(%t1_0, S2MM, 0)
    aie.shim_dma_allocation @outg(%t0_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<3072xi32>, %out: memref<2048xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @outc} : memref<2048xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 1024][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @outg} : memref<2048xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1], packet = <pkt_id = 0, pkt_type = 0>) {id = 1 : i64, issue_token = true, metadata = @inf} : memref<3072xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 1024][1, 1, 1, 1024][0, 0, 0, 1], packet = <pkt_id = 2, pkt_type = 0>) {id = 0 : i64, issue_token = true, metadata = @ing} : memref<3072xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 2048][1, 1, 1, 1024][0, 0, 0, 1], packet = <pkt_id = 1, pkt_type = 0>) {id = 0 : i64, issue_token = true, metadata = @inh} : memref<3072xi32>
      aiex.npu.dma_wait {symbol = @outc}
      aiex.npu.dma_wait {symbol = @outg}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      %0 = aie.amsel<1> (1)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(West : 2, %0)
      %3 = aie.masterset(West : 3, %1)
      aie.packet_rules(East : 2) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 1, %0)
      }
    }
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(West : 2, %0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 2, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<North : 0, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(West : 0, %2)
      %4 = aie.masterset(North : 0, %1)
      %5 = aie.masterset(North : 1, %0)
      aie.packet_rules(East : 3) {
        aie.rule(31, 2, %2)
      }
      aie.packet_rules(East : 2) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 0, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<North : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 0, %0)
      %3 = aie.masterset(North : 1, %1)
      aie.packet_rules(South : 0) {
        aie.rule(31, 1, %0)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %1)
      }
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %1)
      %3 = aie.masterset(DMA : 1, %0)
      aie.packet_rules(South : 0) {
        aie.rule(31, 1, %0)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %1)
      }
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 3, %0)
      aie.packet_rules(East : 0) {
        aie.rule(31, 2, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 0, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 2, %0)
      }
    }
  }
}
