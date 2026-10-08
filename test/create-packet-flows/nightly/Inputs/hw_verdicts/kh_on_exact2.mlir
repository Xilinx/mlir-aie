// HW: PASS
// hops: off
// Two 256-word BDs: the second starts with p0[255], then hdr.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    aie.flow(%t0_0, DMA : 0, %t0_1, DMA : 0)
    aie.packet_flow(5) { aie.packet_source<%t0_1, DMA : 0> aie.packet_dest<%t0_2, DMA : 0> } {keep_pkt_header = true}
    aie.flow(%t0_2, DMA : 0, %t0_0, DMA : 0)
    %mt = aie.buffer(%t0_1) {sym_name = "mt"} : memref<512xi32>
    %mt_p = aie.lock(%t0_1, 0) {init = 2 : i32, sym_name = "mt_p"}
    %mt_c = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "mt_c"}
    %rx = aie.buffer(%t0_2) {sym_name = "rx"} : memref<512xi32>
    %rx_p = aie.lock(%t0_2, 0) {init = 2 : i32, sym_name = "rx_p"}
    %rx_c = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "rx_c"}
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%mt_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%mt_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%mt_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%mt_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end)
      ^c1b0:
        aie.use_lock(%mt_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<512xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
        aie.use_lock(%mt_p, Release, %one)
        aie.next_bd ^c1b1
      ^c1b1:
        aie.use_lock(%mt_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt : memref<512xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
        aie.use_lock(%mt_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%rx_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%rx : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%rx_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%rx_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%rx : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%rx_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end)
      ^c1b0:
        aie.use_lock(%rx_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%rx : memref<512xi32> offset = 0 len = 256)
        aie.use_lock(%rx_p, Release, %one)
        aie.next_bd ^c1b1
      ^c1b1:
        aie.use_lock(%rx_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%rx : memref<512xi32> offset = 256 len = 256)
        aie.use_lock(%rx_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @in0(%t0_0, MM2S, 0)
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<512xi32>, %out: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<512xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @in0} : memref<512xi32>
      aiex.npu.dma_wait {symbol = @out0}
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
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<North : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 5, %0)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 5, %0)
      }
    }
  }
}
