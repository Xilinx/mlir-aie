// HW: HANG
// hops: off
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
    aie.buffer @mt0(%t0_1) : memref<1024xi32>
    aie.lock @mt0_p(%t0_1, 0) {init = 4 : i32}
    aie.lock @mt0_c(%t0_1, 1) {init = 0 : i32}
    aie.buffer @mt1(%t0_1) : memref<1024xi32>
    aie.lock @mt1_p(%t0_1, 2) {init = 4 : i32}
    aie.lock @mt1_c(%t0_1, 3) {init = 0 : i32}
    aie.buffer @mt2(%t0_1) : memref<1024xi32>
    aie.lock @mt2_p(%t0_1, 4) {init = 4 : i32}
    aie.lock @mt2_c(%t0_1, 5) {init = 0 : i32}
    aie.buffer @i11_0(%t1_1) : memref<256xi32>
    aie.lock @i11_0_p(%t1_1, 0) {init = 1 : i32}
    aie.lock @i11_0_c(%t1_1, 1) {init = 0 : i32}
    aie.buffer @i11_1(%t1_1) : memref<256xi32>
    aie.lock @i11_1_p(%t1_1, 2) {init = 1 : i32}
    aie.lock @i11_1_c(%t1_1, 3) {init = 0 : i32}
    aie.buffer @i11_2(%t1_1) : memref<256xi32>
    aie.lock @i11_2_p(%t1_1, 4) {init = 1 : i32}
    aie.lock @i11_2_c(%t1_1, 5) {init = 0 : i32}
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(@mt0_p, AcquireGreaterEqual, %one)
        %mt0 = aie.get_buffer @mt0 : memref<1024xi32>
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(@mt0_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(@mt0_p, AcquireGreaterEqual, %one)
        %mt0_1 = aie.get_buffer @mt0 : memref<1024xi32>
        aie.dma_bd(%mt0_1 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(@mt0_c, Release, %one)
        aie.next_bd ^c0b2
      ^c0b2:
        aie.use_lock(@mt0_p, AcquireGreaterEqual, %one)
        %mt0_2 = aie.get_buffer @mt0 : memref<1024xi32>
        aie.dma_bd(%mt0_2 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(@mt0_c, Release, %one)
        aie.next_bd ^c0b3
      ^c0b3:
        aie.use_lock(@mt0_p, AcquireGreaterEqual, %one)
        %mt0_3 = aie.get_buffer @mt0 : memref<1024xi32>
        aie.dma_bd(%mt0_3 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(@mt0_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(@mt0_c, AcquireGreaterEqual, %one)
        %mt0_4 = aie.get_buffer @mt0 : memref<1024xi32>
        aie.dma_bd(%mt0_4 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(@mt0_p, Release, %one)
        aie.next_bd ^c1b1
      ^c1b1:
        aie.use_lock(@mt0_c, AcquireGreaterEqual, %one)
        %mt0_5 = aie.get_buffer @mt0 : memref<1024xi32>
        aie.dma_bd(%mt0_5 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(@mt0_p, Release, %one)
        aie.next_bd ^c1b2
      ^c1b2:
        aie.use_lock(@mt0_c, AcquireGreaterEqual, %one)
        %mt0_6 = aie.get_buffer @mt0 : memref<1024xi32>
        aie.dma_bd(%mt0_6 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(@mt0_p, Release, %one)
        aie.next_bd ^c1b3
      ^c1b3:
        aie.use_lock(@mt0_c, AcquireGreaterEqual, %one)
        %mt0_7 = aie.get_buffer @mt0 : memref<1024xi32>
        aie.dma_bd(%mt0_7 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(@mt0_p, Release, %one)
        aie.next_bd ^c1b0
      ^s2:
      %d2 = aie.dma_start(S2MM, 1, ^c2b0, ^s3)
      ^c2b0:
        aie.use_lock(@mt1_p, AcquireGreaterEqual, %one)
        %mt1 = aie.get_buffer @mt1 : memref<1024xi32>
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(@mt1_c, Release, %one)
        aie.next_bd ^c2b1
      ^c2b1:
        aie.use_lock(@mt1_p, AcquireGreaterEqual, %one)
        %mt1_1 = aie.get_buffer @mt1 : memref<1024xi32>
        aie.dma_bd(%mt1_1 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(@mt1_c, Release, %one)
        aie.next_bd ^c2b2
      ^c2b2:
        aie.use_lock(@mt1_p, AcquireGreaterEqual, %one)
        %mt1_2 = aie.get_buffer @mt1 : memref<1024xi32>
        aie.dma_bd(%mt1_2 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(@mt1_c, Release, %one)
        aie.next_bd ^c2b3
      ^c2b3:
        aie.use_lock(@mt1_p, AcquireGreaterEqual, %one)
        %mt1_3 = aie.get_buffer @mt1 : memref<1024xi32>
        aie.dma_bd(%mt1_3 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(@mt1_c, Release, %one)
        aie.next_bd ^c2b0
      ^s3:
      %d3 = aie.dma_start(MM2S, 1, ^c3b0, ^s4)
      ^c3b0:
        aie.use_lock(@mt1_c, AcquireGreaterEqual, %one)
        %mt1_4 = aie.get_buffer @mt1 : memref<1024xi32>
        aie.dma_bd(%mt1_4 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(@mt1_p, Release, %one)
        aie.next_bd ^c3b1
      ^c3b1:
        aie.use_lock(@mt1_c, AcquireGreaterEqual, %one)
        %mt1_5 = aie.get_buffer @mt1 : memref<1024xi32>
        aie.dma_bd(%mt1_5 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(@mt1_p, Release, %one)
        aie.next_bd ^c3b2
      ^c3b2:
        aie.use_lock(@mt1_c, AcquireGreaterEqual, %one)
        %mt1_6 = aie.get_buffer @mt1 : memref<1024xi32>
        aie.dma_bd(%mt1_6 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(@mt1_p, Release, %one)
        aie.next_bd ^c3b3
      ^c3b3:
        aie.use_lock(@mt1_c, AcquireGreaterEqual, %one)
        %mt1_7 = aie.get_buffer @mt1 : memref<1024xi32>
        aie.dma_bd(%mt1_7 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(@mt1_p, Release, %one)
        aie.next_bd ^c3b0
      ^s4:
      %d4 = aie.dma_start(S2MM, 2, ^c4b0, ^s5)
      ^c4b0:
        aie.use_lock(@mt2_p, AcquireGreaterEqual, %one)
        %mt2 = aie.get_buffer @mt2 : memref<1024xi32>
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(@mt2_c, Release, %one)
        aie.next_bd ^c4b1
      ^c4b1:
        aie.use_lock(@mt2_p, AcquireGreaterEqual, %one)
        %mt2_1 = aie.get_buffer @mt2 : memref<1024xi32>
        aie.dma_bd(%mt2_1 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(@mt2_c, Release, %one)
        aie.next_bd ^c4b2
      ^c4b2:
        aie.use_lock(@mt2_p, AcquireGreaterEqual, %one)
        %mt2_2 = aie.get_buffer @mt2 : memref<1024xi32>
        aie.dma_bd(%mt2_2 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(@mt2_c, Release, %one)
        aie.next_bd ^c4b3
      ^c4b3:
        aie.use_lock(@mt2_p, AcquireGreaterEqual, %one)
        %mt2_3 = aie.get_buffer @mt2 : memref<1024xi32>
        aie.dma_bd(%mt2_3 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(@mt2_c, Release, %one)
        aie.next_bd ^c4b0
      ^s5:
      %d5 = aie.dma_start(MM2S, 2, ^c5b0, ^end)
      ^c5b0:
        aie.use_lock(@mt2_c, AcquireGreaterEqual, %one)
        %mt2_4 = aie.get_buffer @mt2 : memref<1024xi32>
        aie.dma_bd(%mt2_4 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(@mt2_p, Release, %one)
        aie.next_bd ^c5b1
      ^c5b1:
        aie.use_lock(@mt2_c, AcquireGreaterEqual, %one)
        %mt2_5 = aie.get_buffer @mt2 : memref<1024xi32>
        aie.dma_bd(%mt2_5 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(@mt2_p, Release, %one)
        aie.next_bd ^c5b2
      ^c5b2:
        aie.use_lock(@mt2_c, AcquireGreaterEqual, %one)
        %mt2_6 = aie.get_buffer @mt2 : memref<1024xi32>
        aie.dma_bd(%mt2_6 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(@mt2_p, Release, %one)
        aie.next_bd ^c5b3
      ^c5b3:
        aie.use_lock(@mt2_c, AcquireGreaterEqual, %one)
        %mt2_7 = aie.get_buffer @mt2 : memref<1024xi32>
        aie.dma_bd(%mt2_7 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(@mt2_p, Release, %one)
        aie.next_bd ^c5b0
      ^end:
        aie.end
    }
    %dma_t1_1 = aie.memtile_dma(%t1_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(@i11_0_p, AcquireGreaterEqual, %one)
        %i11_0 = aie.get_buffer @i11_0 : memref<256xi32>
        aie.dma_bd(%i11_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i11_0_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(S2MM, 1, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(@i11_1_p, AcquireGreaterEqual, %one)
        %i11_1 = aie.get_buffer @i11_1 : memref<256xi32>
        aie.dma_bd(%i11_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i11_1_c, Release, %one)
        aie.next_bd ^c1b0
      ^s2:
      %d2 = aie.dma_start(S2MM, 2, ^c2b0, ^s3)
      ^c2b0:
        aie.use_lock(@i11_2_p, AcquireGreaterEqual, %one)
        %i11_2 = aie.get_buffer @i11_2 : memref<256xi32>
        aie.dma_bd(%i11_2 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i11_2_c, Release, %one)
        aie.next_bd ^c2b0
      ^s3:
      %d3 = aie.dma_start(MM2S, 0, ^c3b0, ^end)
      ^c3b0:
        aie.use_lock(@i11_0_c, AcquireGreaterEqual, %one)
        %i11_0_1 = aie.get_buffer @i11_0 : memref<256xi32>
        aie.dma_bd(%i11_0_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i11_0_p, Release, %one)
        aie.next_bd ^c3b1
      ^c3b1:
        aie.use_lock(@i11_1_c, AcquireGreaterEqual, %one)
        %i11_1_1 = aie.get_buffer @i11_1 : memref<256xi32>
        aie.dma_bd(%i11_1_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i11_1_p, Release, %one)
        aie.next_bd ^c3b2
      ^c3b2:
        aie.use_lock(@i11_2_c, AcquireGreaterEqual, %one)
        %i11_2_1 = aie.get_buffer @i11_2 : memref<256xi32>
        aie.dma_bd(%i11_2_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i11_2_p, Release, %one)
        aie.next_bd ^c3b0
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @in0(%t0_0, MM2S, 0)
    aie.shim_dma_allocation @in1(%t0_0, MM2S, 1)
    aie.shim_dma_allocation @in2(%t1_0, MM2S, 0)
    aie.shim_dma_allocation @out(%t1_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<3072xi32>, %out: memref<3072xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @in0} : memref<3072xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 3072][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out} : memref<3072xi32>
      aiex.npu.dma_wait {symbol = @in0}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 1024][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in1} : memref<3072xi32>
      aiex.npu.dma_wait {symbol = @in1}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 2048][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in2} : memref<3072xi32>
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
      %1 = aie.amsel<0> (1)
      %2 = aie.masterset(East : 0, %0, %1)
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
      aie.packet_rules(West : 0) {
        aie.rule(31, 1, %1)
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
      %1 = aie.amsel<1> (0)
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
