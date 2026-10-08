// HW: PASS
// hops: off
// Shim MM2S source, two memcpys with packet info.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t1_0 = aie.tile(1, 0)
    %t0_2 = aie.tile(0, 2)
    aie.packet_flow(5) { aie.packet_source<%t1_0, DMA : 0> aie.packet_dest<%t0_2, DMA : 0> } {keep_pkt_header = true}
    aie.flow(%t0_2, DMA : 0, %t0_0, DMA : 0)
    %rx = aie.buffer(%t0_2) {sym_name = "rx"} : memref<514xi32>
    %rx_p = aie.lock(%t0_2, 0) {init = 1 : i32, sym_name = "rx_p"}
    %rx_c = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "rx_c"}
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%rx_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%rx : memref<514xi32> offset = 0 len = 514)
        aie.use_lock(%rx_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end)
      ^c1b0:
        aie.use_lock(%rx_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%rx : memref<514xi32> offset = 0 len = 514)
        aie.use_lock(%rx_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @in0(%t1_0, MM2S, 0)
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<512xi32>, %out: memref<514xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 514][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<514xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1], packet = <pkt_id = 5, pkt_type = 0>) {id = 0 : i64, issue_token = true, metadata = @in0} : memref<512xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1], packet = <pkt_id = 5, pkt_type = 0>) {id = 1 : i64, issue_token = true, metadata = @in0} : memref<512xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(West : 2, %0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 5, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<North : 0, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 3, %0)
      aie.packet_rules(East : 2) {
        aie.rule(31, 5, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<North : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 3, %0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 5, %0)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 3) {
        aie.rule(31, 5, %0)
      }
    }
  }
}
