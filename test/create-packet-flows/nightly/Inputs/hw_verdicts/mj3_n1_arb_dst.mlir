// HW: PASS
// hops: off
// One block each: fits.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t1_0 = aie.tile(1, 0)
    %t0_1 = aie.tile(0, 1)
    %t1_1 = aie.tile(1, 1)
    aie.flow(%t0_0, DMA : 0, %t0_1, DMA : 0)
    aie.packet_flow(0) { aie.packet_source<%t0_1, DMA : 0> aie.packet_dest<%t1_1, DMA : 0> }
    aie.flow(%t0_0, DMA : 1, %t0_1, DMA : 1)
    aie.packet_flow(1) { aie.packet_source<%t0_1, DMA : 1> aie.packet_dest<%t1_1, DMA : 1> }
    aie.flow(%t1_0, DMA : 0, %t0_1, DMA : 2)
    aie.packet_flow(2) { aie.packet_source<%t0_1, DMA : 2> aie.packet_dest<%t1_1, DMA : 2> }
    aie.flow(%t1_1, DMA : 0, %t1_0, DMA : 0)
    %mt0 = aie.buffer(%t0_1) {sym_name = "mt0"} : memref<256xi32>
    %mt0_p = aie.lock(%t0_1, 0) {init = 1 : i32, sym_name = "mt0_p"}
    %mt0_c = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "mt0_c"}
    %mt1 = aie.buffer(%t0_1) {sym_name = "mt1"} : memref<256xi32>
    %mt1_p = aie.lock(%t0_1, 2) {init = 1 : i32, sym_name = "mt1_p"}
    %mt1_c = aie.lock(%t0_1, 3) {init = 0 : i32, sym_name = "mt1_c"}
    %mt2 = aie.buffer(%t0_1) {sym_name = "mt2"} : memref<256xi32>
    %mt2_p = aie.lock(%t0_1, 4) {init = 1 : i32, sym_name = "mt2_p"}
    %mt2_c = aie.lock(%t0_1, 5) {init = 0 : i32, sym_name = "mt2_c"}
    %i11_0 = aie.buffer(%t1_1) {sym_name = "i11_0"} : memref<256xi32>
    %i11_0_p = aie.lock(%t1_1, 0) {init = 1 : i32, sym_name = "i11_0_p"}
    %i11_0_c = aie.lock(%t1_1, 1) {init = 0 : i32, sym_name = "i11_0_c"}
    %i11_1 = aie.buffer(%t1_1) {sym_name = "i11_1"} : memref<256xi32>
    %i11_1_p = aie.lock(%t1_1, 2) {init = 1 : i32, sym_name = "i11_1_p"}
    %i11_1_c = aie.lock(%t1_1, 3) {init = 0 : i32, sym_name = "i11_1_c"}
    %i11_2 = aie.buffer(%t1_1) {sym_name = "i11_2"} : memref<256xi32>
    %i11_2_p = aie.lock(%t1_1, 4) {init = 1 : i32, sym_name = "i11_2_p"}
    %i11_2_c = aie.lock(%t1_1, 5) {init = 0 : i32, sym_name = "i11_2_c"}
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^end
      ^s2:
      %d2 = aie.dma_start(S2MM, 1, ^c2b0, ^s3)
      ^c2b0:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^end
      ^s3:
      %d3 = aie.dma_start(MM2S, 1, ^c3b0, ^s4)
      ^c3b0:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^end
      ^s4:
      %d4 = aie.dma_start(S2MM, 2, ^c4b0, ^s5)
      ^c4b0:
        aie.use_lock(%mt2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%mt2_c, Release, %one)
        aie.next_bd ^end
      ^s5:
      %d5 = aie.dma_start(MM2S, 2, ^c5b0, ^end)
      ^c5b0:
        aie.use_lock(%mt2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%mt2_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t1_1 = aie.memtile_dma(%t1_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%i11_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(S2MM, 1, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%i11_1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_1_c, Release, %one)
        aie.next_bd ^end
      ^s2:
      %d2 = aie.dma_start(S2MM, 2, ^c2b0, ^s3)
      ^c2b0:
        aie.use_lock(%i11_2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_2 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_2_c, Release, %one)
        aie.next_bd ^end
      ^s3:
      %d3 = aie.dma_start(MM2S, 0, ^c3b0, ^end)
      ^c3b0:
        aie.use_lock(%i11_0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_0_p, Release, %one)
        aie.next_bd ^c3b1
      ^c3b1:
        aie.use_lock(%i11_1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_1_p, Release, %one)
        aie.next_bd ^c3b2
      ^c3b2:
        aie.use_lock(%i11_2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%i11_2 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i11_2_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @in0(%t0_0, MM2S, 0)
    aie.shim_dma_allocation @in1(%t0_0, MM2S, 1)
    aie.shim_dma_allocation @in2(%t1_0, MM2S, 0)
    aie.shim_dma_allocation @out(%t1_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<768xi32>, %out: memref<768xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @in0} : memref<768xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 768][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out} : memref<768xi32>
      aiex.npu.dma_wait {symbol = @in0}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in1} : memref<768xi32>
      aiex.npu.dma_wait {symbol = @in1}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in2} : memref<768xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      aie.connect<East : 2, North : 4>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(East : 0, %0)
      %3 = aie.masterset(East : 2, %1)
      aie.packet_rules(North : 3) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 0, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<South : 4, DMA : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(South : 2, %0)
      %4 = aie.masterset(South : 3, %1)
      %5 = aie.masterset(North : 1, %2)
      aie.packet_rules(DMA : 2) {
        aie.rule(31, 2, %2)
      }
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 3, %0)
      %3 = aie.masterset(North : 5, %1)
      aie.packet_rules(West : 2) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(West : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<DMA : 0, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(DMA : 0, %0)
      %4 = aie.masterset(DMA : 1, %1)
      %5 = aie.masterset(DMA : 2, %2)
      aie.packet_rules(North : 0) {
        aie.rule(31, 2, %2)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(East : 0, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 2, %0)
      }
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 0, %0)
      aie.packet_rules(West : 0) {
        aie.rule(31, 2, %0)
      }
    }
  }
}
