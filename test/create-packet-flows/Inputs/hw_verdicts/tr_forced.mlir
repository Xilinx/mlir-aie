// HW: PASS
// hops: off
// Router output with only arbiter 0 free at (1,0) and (2,0) (gen_forced.py).
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
    %sb_t1_0 = aie.switchbox(%t1_0) {
      %r1_0 = aie.amsel<1> (0)
      %r1_1 = aie.amsel<1> (1)
      %r1_2 = aie.amsel<1> (2)
      %r1_3 = aie.amsel<1> (3)
      %r2_0 = aie.amsel<2> (0)
      %r2_1 = aie.amsel<2> (1)
      %r2_2 = aie.amsel<2> (2)
      %r2_3 = aie.amsel<2> (3)
      %r3_0 = aie.amsel<3> (0)
      %r3_1 = aie.amsel<3> (1)
      %r3_2 = aie.amsel<3> (2)
      %r3_3 = aie.amsel<3> (3)
      %r4_0 = aie.amsel<4> (0)
      %r4_1 = aie.amsel<4> (1)
      %r4_2 = aie.amsel<4> (2)
      %r4_3 = aie.amsel<4> (3)
      %r5_0 = aie.amsel<5> (0)
      %r5_1 = aie.amsel<5> (1)
      %r5_2 = aie.amsel<5> (2)
      %r5_3 = aie.amsel<5> (3)
      %rm1 = aie.masterset(East : 0, %r1_0, %r1_1, %r1_2, %r1_3)
      %rm2 = aie.masterset(East : 1, %r2_0, %r2_1, %r2_2, %r2_3)
      %rm3 = aie.masterset(East : 2, %r3_0, %r3_1, %r3_2, %r3_3)
      %rm4 = aie.masterset(East : 3, %r4_0, %r4_1, %r4_2, %r4_3)
      %rm5 = aie.masterset(South : 4, %r5_0, %r5_1, %r5_2, %r5_3)
    }
    %sb_t2_0 = aie.switchbox(%t2_0) {
      %r1_0 = aie.amsel<1> (0)
      %r1_1 = aie.amsel<1> (1)
      %r1_2 = aie.amsel<1> (2)
      %r1_3 = aie.amsel<1> (3)
      %r2_0 = aie.amsel<2> (0)
      %r2_1 = aie.amsel<2> (1)
      %r2_2 = aie.amsel<2> (2)
      %r2_3 = aie.amsel<2> (3)
      %r3_0 = aie.amsel<3> (0)
      %r3_1 = aie.amsel<3> (1)
      %r3_2 = aie.amsel<3> (2)
      %r3_3 = aie.amsel<3> (3)
      %r4_0 = aie.amsel<4> (0)
      %r4_1 = aie.amsel<4> (1)
      %r4_2 = aie.amsel<4> (2)
      %r4_3 = aie.amsel<4> (3)
      %r5_0 = aie.amsel<5> (0)
      %r5_1 = aie.amsel<5> (1)
      %r5_2 = aie.amsel<5> (2)
      %r5_3 = aie.amsel<5> (3)
      %rm1 = aie.masterset(East : 0, %r1_0, %r1_1, %r1_2, %r1_3)
      %rm2 = aie.masterset(East : 1, %r2_0, %r2_1, %r2_2, %r2_3)
      %rm3 = aie.masterset(East : 2, %r3_0, %r3_1, %r3_2, %r3_3)
      %rm4 = aie.masterset(East : 3, %r4_0, %r4_1, %r4_2, %r4_3)
      %rm5 = aie.masterset(South : 4, %r5_0, %r5_1, %r5_2, %r5_3)
    }
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
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(West : 1, %0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 2, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      %0 = aie.amsel<1> (0)
      %1 = aie.amsel<1> (1)
      %2 = aie.amsel<1> (2)
      %3 = aie.amsel<1> (3)
      %4 = aie.amsel<2> (0)
      %5 = aie.amsel<2> (1)
      %6 = aie.amsel<2> (2)
      %7 = aie.amsel<2> (3)
      %8 = aie.amsel<3> (0)
      %9 = aie.amsel<3> (1)
      %10 = aie.amsel<3> (2)
      %11 = aie.amsel<3> (3)
      %12 = aie.amsel<4> (0)
      %13 = aie.amsel<4> (1)
      %14 = aie.amsel<4> (2)
      %15 = aie.amsel<4> (3)
      %16 = aie.amsel<5> (0)
      %17 = aie.amsel<5> (1)
      %18 = aie.amsel<5> (2)
      %19 = aie.amsel<5> (3)
      %20 = aie.masterset(East : 0, %0, %1, %2, %3)
      %21 = aie.masterset(East : 1, %4, %5, %6, %7)
      %22 = aie.masterset(East : 2, %8, %9, %10, %11)
      %23 = aie.masterset(East : 3, %12, %13, %14, %15)
      %24 = aie.masterset(South : 4, %16, %17, %18, %19)
      aie.connect<North : 0, South : 2>
      %25 = aie.amsel<0> (0)
      %26 = aie.masterset(West : 1, %25)
      aie.packet_rules(South : 3) {
        aie.rule(31, 0, %25)
      }
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      %0 = aie.amsel<1> (0)
      %1 = aie.amsel<1> (1)
      %2 = aie.amsel<1> (2)
      %3 = aie.amsel<1> (3)
      %4 = aie.amsel<2> (0)
      %5 = aie.amsel<2> (1)
      %6 = aie.amsel<2> (2)
      %7 = aie.amsel<2> (3)
      %8 = aie.amsel<3> (0)
      %9 = aie.amsel<3> (1)
      %10 = aie.amsel<3> (2)
      %11 = aie.amsel<3> (3)
      %12 = aie.amsel<4> (0)
      %13 = aie.amsel<4> (1)
      %14 = aie.amsel<4> (2)
      %15 = aie.amsel<4> (3)
      %16 = aie.amsel<5> (0)
      %17 = aie.amsel<5> (1)
      %18 = aie.amsel<5> (2)
      %19 = aie.amsel<5> (3)
      %20 = aie.masterset(East : 0, %0, %1, %2, %3)
      %21 = aie.masterset(East : 1, %4, %5, %6, %7)
      %22 = aie.masterset(East : 2, %8, %9, %10, %11)
      %23 = aie.masterset(East : 3, %12, %13, %14, %15)
      %24 = aie.masterset(South : 4, %16, %17, %18, %19)
      %25 = aie.amsel<0> (0)
      %26 = aie.amsel<0> (1)
      %27 = aie.masterset(North : 1, %26)
      %28 = aie.masterset(North : 2, %25)
      aie.packet_rules(East : 1) {
        aie.rule(31, 2, %26)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 1, %25)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<North : 0, South : 0>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(DMA : 0, %0)
      %4 = aie.masterset(DMA : 1, %1)
      %5 = aie.masterset(West : 3, %2)
      aie.packet_rules(East : 1) {
        aie.rule(31, 2, %2)
      }
      aie.packet_rules(East : 3) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(West : 1) {
        aie.rule(31, 0, %0)
      }
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 3, %0)
      aie.packet_rules(East : 1) {
        aie.rule(31, 0, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 0, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %1)
      %3 = aie.masterset(North : 3, %0)
      aie.packet_rules(North : 2) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 2, %1)
      %3 = aie.masterset(East : 1, %0)
      aie.packet_rules(East : 3) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 0, %0)
      }
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %0)
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 2, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 1, %1)
      }
    }
    %tile_2_2 = aie.tile(2, 2)
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(West : 1, %0)
      %3 = aie.masterset(West : 3, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 2, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 1, %1)
      }
    }
  }
}
