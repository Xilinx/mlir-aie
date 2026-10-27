// HW: PASS
// hops: off
// A and B both into core (0,2) S2MM 0; the router puts them on one link.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    aie.flow(%t0_0, DMA : 0, %t0_1, DMA : 0)
    aie.flow(%t0_0, DMA : 1, %t0_1, DMA : 1)
    aie.packet_flow(0) { aie.packet_source<%t0_1, DMA : 0> aie.packet_dest<%t0_2, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t0_1, DMA : 1> aie.packet_dest<%t0_2, DMA : 0> }
    aie.flow(%t0_2, DMA : 0, %t0_0, DMA : 0)
    aie.buffer @mtA(%t0_1) : memref<1024xi32>
    aie.lock @mtA_p(%t0_1, 0) {init = 4 : i32}
    aie.lock @mtA_c(%t0_1, 1) {init = 0 : i32}
    aie.buffer @mtB(%t0_1) : memref<1024xi32>
    aie.lock @mtB_p(%t0_1, 2) {init = 4 : i32}
    aie.lock @mtB_c(%t0_1, 3) {init = 0 : i32}
    aie.buffer @i02_0(%t0_2) : memref<256xi32>
    aie.lock @i02_0_p(%t0_2, 0) {init = 1 : i32}
    aie.lock @i02_0_c(%t0_2, 1) {init = 0 : i32}
    aie.buffer @o02_0(%t0_2) : memref<256xi32>
    aie.lock @o02_0_p(%t0_2, 2) {init = 1 : i32}
    aie.lock @o02_0_c(%t0_2, 3) {init = 0 : i32}
    %core_t0_2 = aie.core(%t0_2) {
      %one = arith.constant 1 : i32
      %z = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 8 : index
      %cb = arith.constant 256 : index
      aie.use_lock(@o02_0_p, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %cb step %c1 {
        %o02_0 = aie.get_buffer @o02_0 : memref<256xi32>
        memref.store %z, %o02_0[%i] : memref<256xi32>
      }
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(@i02_0_c, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %cb step %c1 {
          %a = arith.addi %i, %c0 : index
          %i02_0 = aie.get_buffer @i02_0 : memref<256xi32>
          %x = memref.load %i02_0[%a] : memref<256xi32>
          %o02_0_1 = aie.get_buffer @o02_0 : memref<256xi32>
          %y = memref.load %o02_0_1[%i] : memref<256xi32>
          %s = arith.addi %x, %y : i32
          memref.store %s, %o02_0_1[%i] : memref<256xi32>
        }
        aie.use_lock(@i02_0_p, Release, %one)
      }
      aie.use_lock(@o02_0_c, Release, %one)
      aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(@mtA_p, AcquireGreaterEqual, %one)
        %mtA = aie.get_buffer @mtA : memref<1024xi32>
        aie.dma_bd(%mtA : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(@mtA_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(@mtA_p, AcquireGreaterEqual, %one)
        %mtA_1 = aie.get_buffer @mtA : memref<1024xi32>
        aie.dma_bd(%mtA_1 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(@mtA_c, Release, %one)
        aie.next_bd ^c0b2
      ^c0b2:
        aie.use_lock(@mtA_p, AcquireGreaterEqual, %one)
        %mtA_2 = aie.get_buffer @mtA : memref<1024xi32>
        aie.dma_bd(%mtA_2 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(@mtA_c, Release, %one)
        aie.next_bd ^c0b3
      ^c0b3:
        aie.use_lock(@mtA_p, AcquireGreaterEqual, %one)
        %mtA_3 = aie.get_buffer @mtA : memref<1024xi32>
        aie.dma_bd(%mtA_3 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(@mtA_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(@mtA_c, AcquireGreaterEqual, %one)
        %mtA_4 = aie.get_buffer @mtA : memref<1024xi32>
        aie.dma_bd(%mtA_4 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(@mtA_p, Release, %one)
        aie.next_bd ^c1b1
      ^c1b1:
        aie.use_lock(@mtA_c, AcquireGreaterEqual, %one)
        %mtA_5 = aie.get_buffer @mtA : memref<1024xi32>
        aie.dma_bd(%mtA_5 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(@mtA_p, Release, %one)
        aie.next_bd ^c1b2
      ^c1b2:
        aie.use_lock(@mtA_c, AcquireGreaterEqual, %one)
        %mtA_6 = aie.get_buffer @mtA : memref<1024xi32>
        aie.dma_bd(%mtA_6 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(@mtA_p, Release, %one)
        aie.next_bd ^c1b3
      ^c1b3:
        aie.use_lock(@mtA_c, AcquireGreaterEqual, %one)
        %mtA_7 = aie.get_buffer @mtA : memref<1024xi32>
        aie.dma_bd(%mtA_7 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(@mtA_p, Release, %one)
        aie.next_bd ^c1b0
      ^s2:
      %d2 = aie.dma_start(S2MM, 1, ^c2b0, ^s3)
      ^c2b0:
        aie.use_lock(@mtB_p, AcquireGreaterEqual, %one)
        %mtB = aie.get_buffer @mtB : memref<1024xi32>
        aie.dma_bd(%mtB : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(@mtB_c, Release, %one)
        aie.next_bd ^c2b1
      ^c2b1:
        aie.use_lock(@mtB_p, AcquireGreaterEqual, %one)
        %mtB_1 = aie.get_buffer @mtB : memref<1024xi32>
        aie.dma_bd(%mtB_1 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(@mtB_c, Release, %one)
        aie.next_bd ^c2b2
      ^c2b2:
        aie.use_lock(@mtB_p, AcquireGreaterEqual, %one)
        %mtB_2 = aie.get_buffer @mtB : memref<1024xi32>
        aie.dma_bd(%mtB_2 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(@mtB_c, Release, %one)
        aie.next_bd ^c2b3
      ^c2b3:
        aie.use_lock(@mtB_p, AcquireGreaterEqual, %one)
        %mtB_3 = aie.get_buffer @mtB : memref<1024xi32>
        aie.dma_bd(%mtB_3 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(@mtB_c, Release, %one)
        aie.next_bd ^c2b0
      ^s3:
      %d3 = aie.dma_start(MM2S, 1, ^c3b0, ^end)
      ^c3b0:
        aie.use_lock(@mtB_c, AcquireGreaterEqual, %one)
        %mtB_4 = aie.get_buffer @mtB : memref<1024xi32>
        aie.dma_bd(%mtB_4 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(@mtB_p, Release, %one)
        aie.next_bd ^c3b1
      ^c3b1:
        aie.use_lock(@mtB_c, AcquireGreaterEqual, %one)
        %mtB_5 = aie.get_buffer @mtB : memref<1024xi32>
        aie.dma_bd(%mtB_5 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(@mtB_p, Release, %one)
        aie.next_bd ^c3b2
      ^c3b2:
        aie.use_lock(@mtB_c, AcquireGreaterEqual, %one)
        %mtB_6 = aie.get_buffer @mtB : memref<1024xi32>
        aie.dma_bd(%mtB_6 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(@mtB_p, Release, %one)
        aie.next_bd ^c3b3
      ^c3b3:
        aie.use_lock(@mtB_c, AcquireGreaterEqual, %one)
        %mtB_7 = aie.get_buffer @mtB : memref<1024xi32>
        aie.dma_bd(%mtB_7 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(@mtB_p, Release, %one)
        aie.next_bd ^c3b0
      ^end:
        aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(@i02_0_p, AcquireGreaterEqual, %one)
        %i02_0_1 = aie.get_buffer @i02_0 : memref<256xi32>
        aie.dma_bd(%i02_0_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i02_0_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end)
      ^c1b0:
        aie.use_lock(@o02_0_c, AcquireGreaterEqual, %one)
        %o02_0_2 = aie.get_buffer @o02_0 : memref<256xi32>
        aie.dma_bd(%o02_0_2 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@o02_0_p, Release, %one)
        aie.next_bd ^c1b0
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @inA(%t0_0, MM2S, 0)
    aie.shim_dma_allocation @inB(%t0_0, MM2S, 1)
    aie.shim_dma_allocation @out(%t0_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<2048xi32>, %out: memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @inA} : memref<2048xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out} : memref<256xi32>
      aiex.npu.dma_wait {symbol = @inA}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 1024][1, 1, 1, 1024][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @inB} : memref<2048xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<North : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 1, %0)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(South : 1) {
        aie.rule(30, 0, %0)
      }
    }
  }
}
