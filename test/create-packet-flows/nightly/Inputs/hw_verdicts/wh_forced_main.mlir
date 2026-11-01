// HW: HANG
module {
  aie.device(npu2) {
    %t1_0 = aie.tile(1, 0)
    %t2_0 = aie.tile(2, 0)
    %t0_2 = aie.tile(0, 2)
    %t1_2 = aie.tile(1, 2)
    %t2_2 = aie.tile(2, 2)
    %t3_2 = aie.tile(3, 2)
    aie.packet_flow(0) { aie.packet_source<%t0_2, DMA : 0> aie.packet_dest<%t2_2, DMA : 0> }
    aie.flow(%t2_2, DMA : 0, %t2_0, DMA : 0)
    aie.packet_flow(1) { aie.packet_source<%t3_2, DMA : 0> aie.packet_dest<%t1_2, DMA : 0> }
    aie.flow(%t1_2, DMA : 0, %t1_0, DMA : 0)
    %g02 = aie.buffer(%t0_2) {sym_name = "g02"} : memref<512xi32>
    %g02_p = aie.lock(%t0_2, 0) {init = 2 : i32, sym_name = "g02_p"}
    %g02_c = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "g02_c"}
    %i22_0 = aie.buffer(%t2_2) {sym_name = "i22_0"} : memref<512xi32>
    %i22_0_p = aie.lock(%t2_2, 0) {init = 2 : i32, sym_name = "i22_0_p"}
    %i22_0_c = aie.lock(%t2_2, 1) {init = 0 : i32, sym_name = "i22_0_c"}
    %o22_0 = aie.buffer(%t2_2) {sym_name = "o22_0"} : memref<256xi32>
    %o22_0_p = aie.lock(%t2_2, 2) {init = 1 : i32, sym_name = "o22_0_p"}
    %o22_0_c = aie.lock(%t2_2, 3) {init = 0 : i32, sym_name = "o22_0_c"}
    %g32 = aie.buffer(%t3_2) {sym_name = "g32"} : memref<512xi32>
    %g32_p = aie.lock(%t3_2, 0) {init = 2 : i32, sym_name = "g32_p"}
    %g32_c = aie.lock(%t3_2, 1) {init = 0 : i32, sym_name = "g32_c"}
    %i12_0 = aie.buffer(%t1_2) {sym_name = "i12_0"} : memref<512xi32>
    %i12_0_p = aie.lock(%t1_2, 0) {init = 2 : i32, sym_name = "i12_0_p"}
    %i12_0_c = aie.lock(%t1_2, 1) {init = 0 : i32, sym_name = "i12_0_c"}
    %o12_0 = aie.buffer(%t1_2) {sym_name = "o12_0"} : memref<256xi32>
    %o12_0_p = aie.lock(%t1_2, 2) {init = 1 : i32, sym_name = "o12_0_p"}
    %o12_0_c = aie.lock(%t1_2, 3) {init = 0 : i32, sym_name = "o12_0_c"}
    %sb_t1_2 = aie.switchbox(%t1_2) {
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
      %rm1 = aie.masterset(North : 0, %r1_0, %r1_1, %r1_2, %r1_3)
      %rm2 = aie.masterset(North : 1, %r2_0, %r2_1, %r2_2, %r2_3)
      %rm3 = aie.masterset(North : 2, %r3_0, %r3_1, %r3_2, %r3_3)
      %rm4 = aie.masterset(North : 3, %r4_0, %r4_1, %r4_2, %r4_3)
      %rm5 = aie.masterset(North : 4, %r5_0, %r5_1, %r5_2, %r5_3)
    }
    %sb_t2_2 = aie.switchbox(%t2_2) {
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
      %rm1 = aie.masterset(North : 0, %r1_0, %r1_1, %r1_2, %r1_3)
      %rm2 = aie.masterset(North : 1, %r2_0, %r2_1, %r2_2, %r2_3)
      %rm3 = aie.masterset(North : 2, %r3_0, %r3_1, %r3_2, %r3_3)
      %rm4 = aie.masterset(North : 3, %r4_0, %r4_1, %r4_2, %r4_3)
      %rm5 = aie.masterset(North : 4, %r5_0, %r5_1, %r5_2, %r5_3)
    }
    %core_t0_2 = aie.core(%t0_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 8 : index
      %cb = arith.constant 256 : index
      %base = arith.constant 1000 : i32
      %k1 = arith.constant 7919 : i32
      %k2 = arith.constant 31 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(%g02_p, AcquireGreaterEqual, %one)
        %dep = arith.constant 2 : index
        %m = arith.remui %it, %dep : index
        %off = arith.muli %m, %cb : index
        %it32 = arith.index_cast %it : index to i32
        %t1 = arith.muli %it32, %k1 : i32
        %b1 = arith.addi %base, %t1 : i32
        scf.for %i = %c0 to %cb step %c1 {
            %i32 = arith.index_cast %i : index to i32
            %t2 = arith.muli %i32, %k2 : i32
            %v = arith.addi %b1, %t2 : i32
            %a = arith.addi %i, %off : index
            memref.store %v, %g02[%a] : memref<512xi32>
        }
        aie.use_lock(%g02_c, Release, %one)
      }
      aie.end
    }
    %core_t2_2 = aie.core(%t2_2) {
      %one = arith.constant 1 : i32
      %z = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 8 : index
      %cb = arith.constant 256 : index
      aie.use_lock(%o22_0_p, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %cb step %c1 {
        memref.store %z, %o22_0[%i] : memref<256xi32>
      }
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(%i22_0_c, AcquireGreaterEqual, %one)
        %dep = arith.constant 2 : index
        %m = arith.remui %it, %dep : index
        %off = arith.muli %m, %cb : index
        scf.for %i = %c0 to %cb step %c1 {
          %a = arith.addi %i, %off : index
          %x = memref.load %i22_0[%a] : memref<512xi32>
          %y = memref.load %o22_0[%i] : memref<256xi32>
          %s = arith.addi %x, %y : i32
          memref.store %s, %o22_0[%i] : memref<256xi32>
        }
        aie.use_lock(%i22_0_p, Release, %one)
      }
      aie.use_lock(%o22_0_c, Release, %one)
      aie.end
    }
    %core_t3_2 = aie.core(%t3_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 8 : index
      %cb = arith.constant 256 : index
      %base = arith.constant 5000 : i32
      %k1 = arith.constant 7919 : i32
      %k2 = arith.constant 31 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(%g32_p, AcquireGreaterEqual, %one)
        %dep = arith.constant 2 : index
        %m = arith.remui %it, %dep : index
        %off = arith.muli %m, %cb : index
        %it32 = arith.index_cast %it : index to i32
        %t1 = arith.muli %it32, %k1 : i32
        %b1 = arith.addi %base, %t1 : i32
        scf.for %i = %c0 to %cb step %c1 {
            %i32 = arith.index_cast %i : index to i32
            %t2 = arith.muli %i32, %k2 : i32
            %v = arith.addi %b1, %t2 : i32
            %a = arith.addi %i, %off : index
            memref.store %v, %g32[%a] : memref<512xi32>
        }
        aie.use_lock(%g32_c, Release, %one)
      }
      aie.end
    }
    %core_t1_2 = aie.core(%t1_2) {
      %one = arith.constant 1 : i32
      %z = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 8 : index
      %cb = arith.constant 256 : index
      aie.use_lock(%o12_0_p, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %cb step %c1 {
        memref.store %z, %o12_0[%i] : memref<256xi32>
      }
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(%i12_0_c, AcquireGreaterEqual, %one)
        %dep = arith.constant 2 : index
        %m = arith.remui %it, %dep : index
        %off = arith.muli %m, %cb : index
        scf.for %i = %c0 to %cb step %c1 {
          %a = arith.addi %i, %off : index
          %x = memref.load %i12_0[%a] : memref<512xi32>
          %y = memref.load %o12_0[%i] : memref<256xi32>
          %s = arith.addi %x, %y : i32
          memref.store %s, %o12_0[%i] : memref<256xi32>
        }
        aie.use_lock(%i12_0_p, Release, %one)
      }
      aie.use_lock(%o12_0_c, Release, %one)
      aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^c0b0, ^end, repeat_count = 3)
      ^c0b0:
        aie.use_lock(%g02_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%g02 : memref<512xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%g02_p, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%g02_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%g02 : memref<512xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%g02_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t2_2 = aie.mem(%t2_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1, repeat_count = 3)
      ^c0b0:
        aie.use_lock(%i22_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i22_0 : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%i22_0_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%i22_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i22_0 : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%i22_0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end)
      ^c1b0:
        aie.use_lock(%o22_0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%o22_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%o22_0_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t3_2 = aie.mem(%t3_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^c0b0, ^end, repeat_count = 3)
      ^c0b0:
        aie.use_lock(%g32_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%g32 : memref<512xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%g32_p, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%g32_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%g32 : memref<512xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%g32_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t1_2 = aie.mem(%t1_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1, repeat_count = 3)
      ^c0b0:
        aie.use_lock(%i12_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i12_0 : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%i12_0_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%i12_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i12_0 : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%i12_0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end)
      ^c1b0:
        aie.use_lock(%o12_0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%o12_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%o12_0_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @out0(%t2_0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t1_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<1xi32>, %out: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<512xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out1} : memref<512xi32>
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(East : 3, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(West : 3, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
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
      %20 = aie.masterset(North : 0, %0, %1, %2, %3)
      %21 = aie.masterset(North : 1, %4, %5, %6, %7)
      %22 = aie.masterset(North : 2, %8, %9, %10, %11)
      %23 = aie.masterset(North : 3, %12, %13, %14, %15)
      %24 = aie.masterset(North : 4, %16, %17, %18, %19)
      aie.connect<DMA : 0, South : 1>
      %25 = aie.amsel<0> (0)
      %26 = aie.amsel<0> (1)
      %27 = aie.masterset(DMA : 0, %26)
      %28 = aie.masterset(East : 1, %25)
      aie.packet_rules(East : 3) {
        aie.rule(31, 1, %26)
      }
      aie.packet_rules(West : 3) {
        aie.rule(31, 0, %25)
      }
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
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
      %20 = aie.masterset(North : 0, %0, %1, %2, %3)
      %21 = aie.masterset(North : 1, %4, %5, %6, %7)
      %22 = aie.masterset(North : 2, %8, %9, %10, %11)
      %23 = aie.masterset(North : 3, %12, %13, %14, %15)
      %24 = aie.masterset(North : 4, %16, %17, %18, %19)
      aie.connect<DMA : 0, South : 1>
      %25 = aie.amsel<0> (0)
      %26 = aie.amsel<0> (1)
      %27 = aie.masterset(DMA : 0, %25)
      %28 = aie.masterset(West : 3, %26)
      aie.packet_rules(East : 3) {
        aie.rule(31, 1, %26)
      }
      aie.packet_rules(West : 1) {
        aie.rule(31, 0, %25)
      }
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<North : 1, South : 2>
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<North : 1, South : 1>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<North : 1, South : 2>
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<North : 1, South : 1>
    }
  }
}
