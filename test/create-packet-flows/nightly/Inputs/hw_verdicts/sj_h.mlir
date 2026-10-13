// HW: PASS
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t1_0 = aie.tile(1, 0)
    %t0_1 = aie.tile(0, 1)
    aie.flow(%t0_0, DMA : 0, %t0_1, DMA : 0)
    aie.packet_flow(0) { aie.packet_source<%t0_1, DMA : 0> aie.packet_dest<%t1_0, DMA : 0> }
    aie.flow(%t0_0, DMA : 1, %t0_1, DMA : 1)
    aie.packet_flow(1) { aie.packet_source<%t0_1, DMA : 1> aie.packet_dest<%t1_0, DMA : 1> }
    %mt0 = aie.buffer(%t0_1) {sym_name = "mt0"} : memref<512xi32>
    %mt0_p = aie.lock(%t0_1, 0) {init = 2 : i32, sym_name = "mt0_p"}
    %mt0_c = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "mt0_c"}
    %mt1 = aie.buffer(%t0_1) {sym_name = "mt1"} : memref<512xi32>
    %mt1_p = aie.lock(%t0_1, 2) {init = 2 : i32, sym_name = "mt1_p"}
    %mt1_c = aie.lock(%t0_1, 3) {init = 0 : i32, sym_name = "mt1_c"}
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<512xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^c1b1
      ^c1b1:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<512xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^end
      ^s2:
      %d2 = aie.dma_start(S2MM, 1, ^c2b0, ^s3)
      ^c2b0:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^c2b1
      ^c2b1:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^end
      ^s3:
      %d3 = aie.dma_start(MM2S, 1, ^c3b0, ^end)
      ^c3b0:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<512xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^c3b1
      ^c3b1:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<512xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @inA(%t0_0, MM2S, 0)
    aie.shim_dma_allocation @inB(%t0_0, MM2S, 1)
    aie.shim_dma_allocation @outA(%t1_0, S2MM, 0)
    aie.shim_dma_allocation @outB(%t1_0, S2MM, 1)
    aie.runtime_sequence(%in: memref<1024xi32>, %out: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @inA} : memref<1024xi32>
      aiex.npu.dma_wait {symbol = @inA}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 512][1, 1, 1, 512][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @inB} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 512][1, 1, 1, 512][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @outB} : memref<1024xi32>
      aiex.npu.dma_wait {symbol = @outB}
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @outA} : memref<1024xi32>
      aiex.npu.dma_wait {symbol = @outA}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<North : 3, DMA : 1>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 2, %0)
      %3 = aie.masterset(South : 3, %1)
      aie.packet_rules(West : 1) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(West : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(East : 0, %1)
      %3 = aie.masterset(East : 1, %0)
      aie.packet_rules(North : 1) {
        aie.rule(31, 1, %0)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 0, %1)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 1, %1)
      %3 = aie.masterset(South : 2, %0)
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
  }
}
