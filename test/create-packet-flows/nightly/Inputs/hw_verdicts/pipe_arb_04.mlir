// HW: HANG
// hops: off
// Flow 0 into (0,4) and flow 1 out of it.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t1_0 = aie.tile(1, 0)
    %t0_4 = aie.tile(0, 4)
    %t0_5 = aie.tile(0, 5)
    aie.packet_flow(0) { aie.packet_source<%t0_0, DMA : 0> aie.packet_dest<%t0_4, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t0_4, DMA : 0> aie.packet_dest<%t0_5, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t0_5, DMA : 0> aie.packet_dest<%t1_0, DMA : 0> }
    %i04_0 = aie.buffer(%t0_4) {sym_name = "i04_0"} : memref<256xi32>
    %i04_0_p = aie.lock(%t0_4, 0) {init = 1 : i32, sym_name = "i04_0_p"}
    %i04_0_c = aie.lock(%t0_4, 1) {init = 0 : i32, sym_name = "i04_0_c"}
    %o04 = aie.buffer(%t0_4) {sym_name = "o04"} : memref<256xi32>
    %o04_p = aie.lock(%t0_4, 2) {init = 1 : i32, sym_name = "o04_p"}
    %o04_c = aie.lock(%t0_4, 3) {init = 0 : i32, sym_name = "o04_c"}
    %i05_0 = aie.buffer(%t0_5) {sym_name = "i05_0"} : memref<256xi32>
    %i05_0_p = aie.lock(%t0_5, 0) {init = 1 : i32, sym_name = "i05_0_p"}
    %i05_0_c = aie.lock(%t0_5, 1) {init = 0 : i32, sym_name = "i05_0_c"}
    %o05 = aie.buffer(%t0_5) {sym_name = "o05"} : memref<256xi32>
    %o05_p = aie.lock(%t0_5, 2) {init = 1 : i32, sym_name = "o05_p"}
    %o05_c = aie.lock(%t0_5, 3) {init = 0 : i32, sym_name = "o05_c"}
    %core_t0_4 = aie.core(%t0_4) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 8 : index
      %cb = arith.constant 256 : index
      %k = arith.constant 1 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(%i04_0_c, AcquireGreaterEqual, %one)
        aie.use_lock(%o04_p, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %cb step %c1 {
          %a0 = arith.addi %i, %c0 : index
          %x0 = memref.load %i04_0[%a0] : memref<256xi32>
          %s0 = arith.addi %k, %x0 : i32
          %ao = arith.addi %i, %c0 : index
          memref.store %s0, %o04[%ao] : memref<256xi32>
        }
        aie.use_lock(%i04_0_p, Release, %one)
        aie.use_lock(%o04_c, Release, %one)
      }
      aie.end
    }
    %core_t0_5 = aie.core(%t0_5) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 8 : index
      %cb = arith.constant 256 : index
      %k = arith.constant 2 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(%i05_0_c, AcquireGreaterEqual, %one)
        aie.use_lock(%o05_p, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %cb step %c1 {
          %a0 = arith.addi %i, %c0 : index
          %x0 = memref.load %i05_0[%a0] : memref<256xi32>
          %s0 = arith.addi %k, %x0 : i32
          %ao = arith.addi %i, %c0 : index
          memref.store %s0, %o05[%ao] : memref<256xi32>
        }
        aie.use_lock(%i05_0_p, Release, %one)
        aie.use_lock(%o05_c, Release, %one)
      }
      aie.end
    }
    %dma_t0_4 = aie.mem(%t0_4) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%i04_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i04_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i04_0_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end)
      ^c1b0:
        aie.use_lock(%o04_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%o04 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%o04_p, Release, %one)
        aie.next_bd ^c1b0
      ^end:
        aie.end
    }
    %dma_t0_5 = aie.mem(%t0_5) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%i05_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i05_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i05_0_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end)
      ^c1b0:
        aie.use_lock(%o05_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%o05 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%o05_p, Release, %one)
        aie.next_bd ^c1b0
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @in0(%t0_0, MM2S, 0)
    aie.shim_dma_allocation @out(%t1_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<2048xi32>, %out: memref<2048xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1], packet = <pkt_id = 0, pkt_type = 0>) {id = 0 : i64, issue_token = true, metadata = @in0} : memref<2048xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out} : memref<2048xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 0, %0)
      }
    }
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 2, %0)
      aie.packet_rules(North : 1) {
        aie.rule(31, 2, %0)
      }
    }
    %tile_0_4 = aie.tile(0, 4)
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      %0 = aie.amsel<1> (1)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(DMA : 0, %1)
      %4 = aie.masterset(South : 2, %2)
      %5 = aie.masterset(North : 3, %0)
      aie.packet_rules(North : 0) {
        aie.rule(31, 2, %2)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }
      aie.packet_rules(South : 4) {
        aie.rule(31, 0, %1)
      }
    }
    %tile_0_5 = aie.tile(0, 5)
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %1)
      %3 = aie.masterset(South : 0, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 2, %0)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 1, %1)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 0, %0)
      %3 = aie.masterset(East : 3, %1)
      aie.packet_rules(North : 3) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 3, %1)
      %3 = aie.masterset(North : 4, %0)
      aie.packet_rules(North : 2) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(South : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 1, %0)
      aie.packet_rules(North : 1) {
        aie.rule(31, 2, %0)
      }
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 1, %0)
      aie.packet_rules(West : 3) {
        aie.rule(31, 2, %0)
      }
    }
  }
}
