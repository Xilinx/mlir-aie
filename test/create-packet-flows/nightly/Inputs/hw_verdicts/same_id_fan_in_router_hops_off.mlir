// HW: PASS
// hops: off
// router_hw.py same_id_fan_in, router_hops_off: the same, with no packet hop circuit switched. Observed on NPU Strix: 20 PASS.
module {
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
// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 2, %0)
      %2 = aie.masterset(West : 3, %0)
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 24, %0)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 24, %0)
      }
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<North : 1, South : 2>
      aie.connect<North : 0, South : 3>
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(North : 3) {
        aie.rule(31, 24, %0)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 24, %0)
      }
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<North : 0, South : 0>
      aie.connect<DMA : 0, East : 3>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %1)
      %3 = aie.masterset(South : 2, %0)
      %4 = aie.masterset(South : 3, %1)
      aie.packet_rules(North : 3) {
        aie.rule(31, 24, %1)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 24, %0)
      }
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %0)
      %3 = aie.masterset(South : 2, %0)
      %4 = aie.masterset(South : 3, %1)
      aie.packet_rules(North : 2) {
        aie.rule(31, 24, %1)
      }
      aie.packet_rules(West : 1) {
        aie.rule(31, 24, %0)
      }
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<North : 2, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<North : 2, South : 2>
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<North : 3, South : 2>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<North : 1, South : 3>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(East : 2, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 24, %0)
      }
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<DMA : 0, South : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(East : 0) {
        aie.rule(31, 24, %0)
      }
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_3_1 = aie.tile(3, 1)
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<North : 0, South : 0>
    }
    %tile_3_2 = aie.tile(3, 2)
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<West : 3, South : 0>
    }
    %tile_1_3 = aie.tile(1, 3)
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(East : 1, %0)
      aie.packet_rules(West : 2) {
        aie.rule(31, 24, %0)
      }
    }
    %tile_1_4 = aie.tile(1, 4)
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(West : 0, %0)
      aie.packet_rules(East : 3) {
        aie.rule(31, 24, %0)
      }
    }
  }
}
