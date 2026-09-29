// HW: PASS
// hops: off
// cautious: passes on HW on the fabric's few words of buffering; ovr_6_arb hangs
// A overruns its window by 4 words; A and B share the core's arbiter.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    aie.flow(%t0_0, DMA : 0, %t0_1, DMA : 0)
    aie.packet_flow(0) { aie.packet_source<%t0_1, DMA : 0> aie.packet_dest<%t0_2, DMA : 0> }
    aie.flow(%t0_0, DMA : 1, %t0_1, DMA : 1)
    aie.packet_flow(1) { aie.packet_source<%t0_1, DMA : 1> aie.packet_dest<%t0_2, DMA : 1> }
    aie.flow(%t0_2, DMA : 0, %t0_0, DMA : 0)
    %mt0 = aie.buffer(%t0_1) {sym_name = "mt0"} : memref<260xi32>
    %mt0_p = aie.lock(%t0_1, 0) {init = 1 : i32, sym_name = "mt0_p"}
    %mt0_c = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "mt0_c"}
    %mt1 = aie.buffer(%t0_1) {sym_name = "mt1"} : memref<256xi32>
    %mt1_p = aie.lock(%t0_1, 2) {init = 1 : i32, sym_name = "mt1_p"}
    %mt1_c = aie.lock(%t0_1, 3) {init = 0 : i32, sym_name = "mt1_c"}
    %i02_0 = aie.buffer(%t0_2) {sym_name = "i02_0"} : memref<256xi32>
    %i02_0_p = aie.lock(%t0_2, 0) {init = 1 : i32, sym_name = "i02_0_p"}
    %i02_0_c = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "i02_0_c"}
    %i02_1 = aie.buffer(%t0_2) {sym_name = "i02_1"} : memref<256xi32>
    %i02_1_p = aie.lock(%t0_2, 2) {init = 1 : i32, sym_name = "i02_1_p"}
    %i02_1_c = aie.lock(%t0_2, 3) {init = 0 : i32, sym_name = "i02_1_c"}
    %o02_0 = aie.buffer(%t0_2) {sym_name = "o02_0"} : memref<256xi32>
    %o02_0_p = aie.lock(%t0_2, 4) {init = 1 : i32, sym_name = "o02_0_p"}
    %o02_0_c = aie.lock(%t0_2, 5) {init = 0 : i32, sym_name = "o02_0_c"}
    %core_t0_2 = aie.core(%t0_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 1 : index
      %cb = arith.constant 256 : index
      %k = arith.constant 0 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(%i02_0_c, AcquireGreaterEqual, %one)
        aie.use_lock(%i02_1_c, AcquireGreaterEqual, %one)
        aie.use_lock(%o02_0_p, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %cb step %c1 {
          %a0 = arith.addi %i, %c0 : index
          %x0 = memref.load %i02_0[%a0] : memref<256xi32>
          %s0 = arith.addi %k, %x0 : i32
          %a1 = arith.addi %i, %c0 : index
          %x1 = memref.load %i02_1[%a1] : memref<256xi32>
          %s1 = arith.addi %s0, %x1 : i32
          %ao = arith.addi %i, %c0 : index
          memref.store %s1, %o02_0[%ao] : memref<256xi32>
        }
        aie.use_lock(%i02_0_p, Release, %one)
        aie.use_lock(%i02_1_p, Release, %one)
        aie.use_lock(%o02_0_c, Release, %one)
      }
      aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<260xi32> offset = 0 len = 260)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<260xi32> offset = 0 len = 260) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
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
      %d3 = aie.dma_start(MM2S, 1, ^c3b0, ^end)
      ^c3b0:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%i02_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i02_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i02_0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(S2MM, 1, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%i02_1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i02_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i02_1_c, Release, %one)
        aie.next_bd ^end
      ^s2:
      %d2 = aie.dma_start(MM2S, 0, ^c2b0, ^end)
      ^c2b0:
        aie.use_lock(%o02_0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%o02_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%o02_0_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @inA(%t0_0, MM2S, 0)
    aie.shim_dma_allocation @inB(%t0_0, MM2S, 1)
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<516xi32>, %out: memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 260][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @inA} : memref<516xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out0} : memref<256xi32>
      aiex.npu.dma_wait {symbol = @inA}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 260][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @inB} : memref<516xi32>
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
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 0, %1)
      %3 = aie.masterset(North : 1, %0)
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<1> (1)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %1)
      %3 = aie.masterset(DMA : 1, %0)
      aie.packet_rules(South : 0) {
        aie.rule(31, 1, %0)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %1)
      }
    }
  }
}
