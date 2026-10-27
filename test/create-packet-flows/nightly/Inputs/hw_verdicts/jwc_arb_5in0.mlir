// HW: PASS
// hops: off
// id 5 (one block, fits S2MM 5) on join input 0's arbiter.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    %t1_2 = aie.tile(1, 2)
    %t2_2 = aie.tile(2, 2)
    %t3_2 = aie.tile(3, 2)
    %t4_2 = aie.tile(4, 2)
    %t5_2 = aie.tile(5, 2)
    %t0_3 = aie.tile(0, 3)
    aie.packet_flow(0) { aie.packet_source<%t0_2, DMA : 0> aie.packet_dest<%t0_1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t1_2, DMA : 0> aie.packet_dest<%t0_1, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t2_2, DMA : 0> aie.packet_dest<%t0_1, DMA : 2> }
    aie.packet_flow(3) { aie.packet_source<%t3_2, DMA : 0> aie.packet_dest<%t0_1, DMA : 3> }
    aie.packet_flow(4) { aie.packet_source<%t4_2, DMA : 0> aie.packet_dest<%t0_1, DMA : 4> }
    aie.packet_flow(10) { aie.packet_source<%t0_1, DMA : 0> aie.packet_dest<%t0_3, DMA : 0> }
    aie.flow(%t0_3, DMA : 0, %t0_0, DMA : 0)
    aie.packet_flow(5) { aie.packet_source<%t5_2, DMA : 0> aie.packet_dest<%t0_1, DMA : 5> }
    aie.flow(%t0_1, DMA : 1, %t0_0, DMA : 1)
    aie.buffer @g0(%t0_2) : memref<256xi32>
    aie.lock @g0_p(%t0_2, 0) {init = 1 : i32}
    aie.lock @g0_c(%t0_2, 1) {init = 0 : i32}
    aie.buffer @g1(%t1_2) : memref<256xi32>
    aie.lock @g1_p(%t1_2, 0) {init = 1 : i32}
    aie.lock @g1_c(%t1_2, 1) {init = 0 : i32}
    aie.lock @spin_t1_2(%t1_2, 2) {init = 1 : i32}
    aie.buffer @g2(%t2_2) : memref<256xi32>
    aie.lock @g2_p(%t2_2, 0) {init = 1 : i32}
    aie.lock @g2_c(%t2_2, 1) {init = 0 : i32}
    aie.buffer @g3(%t3_2) : memref<256xi32>
    aie.lock @g3_p(%t3_2, 0) {init = 1 : i32}
    aie.lock @g3_c(%t3_2, 1) {init = 0 : i32}
    aie.buffer @g4(%t4_2) : memref<256xi32>
    aie.lock @g4_p(%t4_2, 0) {init = 1 : i32}
    aie.lock @g4_c(%t4_2, 1) {init = 0 : i32}
    aie.buffer @i01_0(%t0_1) : memref<256xi32>
    aie.lock @i01_0_p(%t0_1, 0) {init = 1 : i32}
    aie.lock @i01_0_c(%t0_1, 1) {init = 0 : i32}
    aie.buffer @i01_1(%t0_1) : memref<256xi32>
    aie.lock @i01_1_p(%t0_1, 2) {init = 1 : i32}
    aie.lock @i01_1_c(%t0_1, 3) {init = 0 : i32}
    aie.buffer @i01_2(%t0_1) : memref<256xi32>
    aie.lock @i01_2_p(%t0_1, 4) {init = 1 : i32}
    aie.lock @i01_2_c(%t0_1, 5) {init = 0 : i32}
    aie.buffer @i01_3(%t0_1) : memref<256xi32>
    aie.lock @i01_3_p(%t0_1, 6) {init = 1 : i32}
    aie.lock @i01_3_c(%t0_1, 7) {init = 0 : i32}
    aie.buffer @i01_4(%t0_1) : memref<256xi32>
    aie.lock @i01_4_p(%t0_1, 8) {init = 1 : i32}
    aie.lock @i01_4_c(%t0_1, 9) {init = 0 : i32}
    aie.buffer @i03_0(%t0_3) : memref<256xi32>
    aie.lock @i03_0_p(%t0_3, 0) {init = 1 : i32}
    aie.lock @i03_0_c(%t0_3, 1) {init = 0 : i32}
    aie.buffer @o03_0(%t0_3) : memref<256xi32>
    aie.lock @o03_0_p(%t0_3, 2) {init = 1 : i32}
    aie.lock @o03_0_c(%t0_3, 3) {init = 0 : i32}
    aie.buffer @g5(%t5_2) : memref<256xi32>
    aie.lock @g5_p(%t5_2, 0) {init = 1 : i32}
    aie.lock @g5_c(%t5_2, 1) {init = 0 : i32}
    aie.buffer @r5(%t0_1) : memref<256xi32>
    aie.lock @r5_p(%t0_1, 10) {init = 1 : i32}
    aie.lock @r5_c(%t0_1, 11) {init = 0 : i32}
    %core_t0_2 = aie.core(%t0_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cb = arith.constant 256 : index
      %base = arith.constant 17 : i32
      %k1 = arith.constant 7919 : i32
      %k2 = arith.constant 31 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(@g0_p, AcquireGreaterEqual, %one)
        %it32 = arith.index_cast %it : index to i32
        %t1 = arith.muli %it32, %k1 : i32
        %b1 = arith.addi %base, %t1 : i32
        scf.for %i = %c0 to %cb step %c1 {
            %i32 = arith.index_cast %i : index to i32
            %t2 = arith.muli %i32, %k2 : i32
            %v = arith.addi %b1, %t2 : i32
            %a = arith.addi %i, %c0 : index
            %g0 = aie.get_buffer @g0 : memref<256xi32>
            memref.store %v, %g0[%a] : memref<256xi32>
        }
        aie.use_lock(@g0_c, Release, %one)
      }
      aie.end
    }
    %core_t1_2 = aie.core(%t1_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cb = arith.constant 256 : index
      %base = arith.constant 1000020 : i32
      %k1 = arith.constant 7919 : i32
      %k2 = arith.constant 31 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(@g1_p, AcquireGreaterEqual, %one)
        %it32 = arith.index_cast %it : index to i32
        %t1 = arith.muli %it32, %k1 : i32
        %b1 = arith.addi %base, %t1 : i32
        scf.for %i = %c0 to %cb step %c1 {
            %i32 = arith.index_cast %i : index to i32
            %t2 = arith.muli %i32, %k2 : i32
            %v = arith.addi %b1, %t2 : i32
            %a = arith.addi %i, %c0 : index
            %g1 = aie.get_buffer @g1 : memref<256xi32>
            memref.store %v, %g1[%a] : memref<256xi32>
        }
        %cspin = arith.constant 2000 : index
        scf.for %sp = %c0 to %cspin step %c1 {
          aie.use_lock(@spin_t1_2, AcquireGreaterEqual, %one)
          aie.use_lock(@spin_t1_2, Release, %one)
        }
        aie.use_lock(@g1_c, Release, %one)
      }
      aie.end
    }
    %core_t2_2 = aie.core(%t2_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cb = arith.constant 256 : index
      %base = arith.constant 2000023 : i32
      %k1 = arith.constant 7919 : i32
      %k2 = arith.constant 31 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(@g2_p, AcquireGreaterEqual, %one)
        %it32 = arith.index_cast %it : index to i32
        %t1 = arith.muli %it32, %k1 : i32
        %b1 = arith.addi %base, %t1 : i32
        scf.for %i = %c0 to %cb step %c1 {
            %i32 = arith.index_cast %i : index to i32
            %t2 = arith.muli %i32, %k2 : i32
            %v = arith.addi %b1, %t2 : i32
            %a = arith.addi %i, %c0 : index
            %g2 = aie.get_buffer @g2 : memref<256xi32>
            memref.store %v, %g2[%a] : memref<256xi32>
        }
        aie.use_lock(@g2_c, Release, %one)
      }
      aie.end
    }
    %core_t3_2 = aie.core(%t3_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cb = arith.constant 256 : index
      %base = arith.constant 3000026 : i32
      %k1 = arith.constant 7919 : i32
      %k2 = arith.constant 31 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(@g3_p, AcquireGreaterEqual, %one)
        %it32 = arith.index_cast %it : index to i32
        %t1 = arith.muli %it32, %k1 : i32
        %b1 = arith.addi %base, %t1 : i32
        scf.for %i = %c0 to %cb step %c1 {
            %i32 = arith.index_cast %i : index to i32
            %t2 = arith.muli %i32, %k2 : i32
            %v = arith.addi %b1, %t2 : i32
            %a = arith.addi %i, %c0 : index
            %g3 = aie.get_buffer @g3 : memref<256xi32>
            memref.store %v, %g3[%a] : memref<256xi32>
        }
        aie.use_lock(@g3_c, Release, %one)
      }
      aie.end
    }
    %core_t4_2 = aie.core(%t4_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cb = arith.constant 256 : index
      %base = arith.constant 4000029 : i32
      %k1 = arith.constant 7919 : i32
      %k2 = arith.constant 31 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(@g4_p, AcquireGreaterEqual, %one)
        %it32 = arith.index_cast %it : index to i32
        %t1 = arith.muli %it32, %k1 : i32
        %b1 = arith.addi %base, %t1 : i32
        scf.for %i = %c0 to %cb step %c1 {
            %i32 = arith.index_cast %i : index to i32
            %t2 = arith.muli %i32, %k2 : i32
            %v = arith.addi %b1, %t2 : i32
            %a = arith.addi %i, %c0 : index
            %g4 = aie.get_buffer @g4 : memref<256xi32>
            memref.store %v, %g4[%a] : memref<256xi32>
        }
        aie.use_lock(@g4_c, Release, %one)
      }
      aie.end
    }
    %core_t0_3 = aie.core(%t0_3) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 20 : index
      %cb = arith.constant 256 : index
      %k = arith.constant 0 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(@i03_0_c, AcquireGreaterEqual, %one)
        aie.use_lock(@o03_0_p, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %cb step %c1 {
          %a0 = arith.addi %i, %c0 : index
          %i03_0 = aie.get_buffer @i03_0 : memref<256xi32>
          %x0 = memref.load %i03_0[%a0] : memref<256xi32>
          %s0 = arith.addi %k, %x0 : i32
          %ao = arith.addi %i, %c0 : index
          %o03_0 = aie.get_buffer @o03_0 : memref<256xi32>
          memref.store %s0, %o03_0[%ao] : memref<256xi32>
        }
        aie.use_lock(@i03_0_p, Release, %one)
        aie.use_lock(@o03_0_c, Release, %one)
      }
      aie.end
    }
    %core_t5_2 = aie.core(%t5_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 1 : index
      %cb = arith.constant 256 : index
      %base = arith.constant 5000032 : i32
      %k1 = arith.constant 7919 : i32
      %k2 = arith.constant 31 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(@g5_p, AcquireGreaterEqual, %one)
        %it32 = arith.index_cast %it : index to i32
        %t1 = arith.muli %it32, %k1 : i32
        %b1 = arith.addi %base, %t1 : i32
        scf.for %i = %c0 to %cb step %c1 {
            %i32 = arith.index_cast %i : index to i32
            %t2 = arith.muli %i32, %k2 : i32
            %v = arith.addi %b1, %t2 : i32
            %a = arith.addi %i, %c0 : index
            %g5 = aie.get_buffer @g5 : memref<256xi32>
            memref.store %v, %g5[%a] : memref<256xi32>
        }
        aie.use_lock(@g5_c, Release, %one)
      }
      aie.end
    }
    %dma_t0_2 = aie.mem(%t0_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^c0b0, ^end, repeat_count = 3)
      ^c0b0:
        aie.use_lock(@g0_c, AcquireGreaterEqual, %one)
        %g0_1 = aie.get_buffer @g0 : memref<256xi32>
        aie.dma_bd(%g0_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(@g0_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t1_2 = aie.mem(%t1_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^c0b0, ^end, repeat_count = 3)
      ^c0b0:
        aie.use_lock(@g1_c, AcquireGreaterEqual, %one)
        %g1_1 = aie.get_buffer @g1 : memref<256xi32>
        aie.dma_bd(%g1_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(@g1_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t2_2 = aie.mem(%t2_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^c0b0, ^end, repeat_count = 3)
      ^c0b0:
        aie.use_lock(@g2_c, AcquireGreaterEqual, %one)
        %g2_1 = aie.get_buffer @g2 : memref<256xi32>
        aie.dma_bd(%g2_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(@g2_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t3_2 = aie.mem(%t3_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^c0b0, ^end, repeat_count = 3)
      ^c0b0:
        aie.use_lock(@g3_c, AcquireGreaterEqual, %one)
        %g3_1 = aie.get_buffer @g3 : memref<256xi32>
        aie.dma_bd(%g3_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
        aie.use_lock(@g3_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t4_2 = aie.mem(%t4_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^c0b0, ^end, repeat_count = 3)
      ^c0b0:
        aie.use_lock(@g4_c, AcquireGreaterEqual, %one)
        %g4_1 = aie.get_buffer @g4 : memref<256xi32>
        aie.dma_bd(%g4_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
        aie.use_lock(@g4_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1, repeat_count = 3)
      ^c0b0:
        aie.use_lock(@i01_0_p, AcquireGreaterEqual, %one)
        %i01_0 = aie.get_buffer @i01_0 : memref<256xi32>
        aie.dma_bd(%i01_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i01_0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(S2MM, 1, ^c1b0, ^s2, repeat_count = 3)
      ^c1b0:
        aie.use_lock(@i01_1_p, AcquireGreaterEqual, %one)
        %i01_1 = aie.get_buffer @i01_1 : memref<256xi32>
        aie.dma_bd(%i01_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i01_1_c, Release, %one)
        aie.next_bd ^end
      ^s2:
      %d2 = aie.dma_start(S2MM, 2, ^c2b0, ^s3, repeat_count = 3)
      ^c2b0:
        aie.use_lock(@i01_2_p, AcquireGreaterEqual, %one)
        %i01_2 = aie.get_buffer @i01_2 : memref<256xi32>
        aie.dma_bd(%i01_2 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i01_2_c, Release, %one)
        aie.next_bd ^end
      ^s3:
      %d3 = aie.dma_start(S2MM, 3, ^c3b0, ^s4, repeat_count = 3)
      ^c3b0:
        aie.use_lock(@i01_3_p, AcquireGreaterEqual, %one)
        %i01_3 = aie.get_buffer @i01_3 : memref<256xi32>
        aie.dma_bd(%i01_3 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i01_3_c, Release, %one)
        aie.next_bd ^end
      ^s4:
      %d4 = aie.dma_start(S2MM, 4, ^c4b0, ^s5, repeat_count = 3)
      ^c4b0:
        aie.use_lock(@i01_4_p, AcquireGreaterEqual, %one)
        %i01_4 = aie.get_buffer @i01_4 : memref<256xi32>
        aie.dma_bd(%i01_4 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i01_4_c, Release, %one)
        aie.next_bd ^end
      ^s5:
      %d5 = aie.dma_start(MM2S, 0, ^c5b0, ^s6, repeat_count = 3)
      ^c5b0:
        aie.use_lock(@i01_0_c, AcquireGreaterEqual, %one)
        %i01_0_1 = aie.get_buffer @i01_0 : memref<256xi32>
        aie.dma_bd(%i01_0_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
        aie.use_lock(@i01_0_p, Release, %one)
        aie.next_bd ^c5b1
      ^c5b1:
        aie.use_lock(@i01_1_c, AcquireGreaterEqual, %one)
        %i01_1_1 = aie.get_buffer @i01_1 : memref<256xi32>
        aie.dma_bd(%i01_1_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
        aie.use_lock(@i01_1_p, Release, %one)
        aie.next_bd ^c5b2
      ^c5b2:
        aie.use_lock(@i01_2_c, AcquireGreaterEqual, %one)
        %i01_2_1 = aie.get_buffer @i01_2 : memref<256xi32>
        aie.dma_bd(%i01_2_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
        aie.use_lock(@i01_2_p, Release, %one)
        aie.next_bd ^c5b3
      ^c5b3:
        aie.use_lock(@i01_3_c, AcquireGreaterEqual, %one)
        %i01_3_1 = aie.get_buffer @i01_3 : memref<256xi32>
        aie.dma_bd(%i01_3_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
        aie.use_lock(@i01_3_p, Release, %one)
        aie.next_bd ^c5b4
      ^c5b4:
        aie.use_lock(@i01_4_c, AcquireGreaterEqual, %one)
        %i01_4_1 = aie.get_buffer @i01_4 : memref<256xi32>
        aie.dma_bd(%i01_4_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
        aie.use_lock(@i01_4_p, Release, %one)
        aie.next_bd ^end
      ^s6:
      %d6 = aie.dma_start(S2MM, 5, ^c6b0, ^s7)
      ^c6b0:
        aie.use_lock(@r5_p, AcquireGreaterEqual, %one)
        %r5 = aie.get_buffer @r5 : memref<256xi32>
        aie.dma_bd(%r5 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@r5_c, Release, %one)
        aie.next_bd ^end
      ^s7:
      %d7 = aie.dma_start(MM2S, 1, ^c7b0, ^end)
      ^c7b0:
        aie.use_lock(@r5_c, AcquireGreaterEqual, %one)
        %r5_1 = aie.get_buffer @r5 : memref<256xi32>
        aie.dma_bd(%r5_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@r5_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t0_3 = aie.mem(%t0_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1, repeat_count = 19)
      ^c0b0:
        aie.use_lock(@i03_0_p, AcquireGreaterEqual, %one)
        %i03_0_1 = aie.get_buffer @i03_0 : memref<256xi32>
        aie.dma_bd(%i03_0_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@i03_0_c, Release, %one)
        aie.next_bd ^end
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^end, repeat_count = 19)
      ^c1b0:
        aie.use_lock(@o03_0_c, AcquireGreaterEqual, %one)
        %o03_0_1 = aie.get_buffer @o03_0 : memref<256xi32>
        aie.dma_bd(%o03_0_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(@o03_0_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %dma_t5_2 = aie.mem(%t5_2) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^c0b0, ^end)
      ^c0b0:
        aie.use_lock(@g5_c, AcquireGreaterEqual, %one)
        %g5_1 = aie.get_buffer @g5 : memref<256xi32>
        aie.dma_bd(%g5_1 : memref<256xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
        aie.use_lock(@g5_p, Release, %one)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t0_0, S2MM, 1)
    aie.runtime_sequence(%in: memref<1xi32>, %out: memref<5376xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 5120][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<5376xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 5120][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out1} : memref<5376xi32>
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.amsel<4> (0)
      %5 = aie.masterset(South : 0, %3)
      %6 = aie.masterset(South : 1, %2)
      %7 = aie.masterset(South : 2, %4)
      %8 = aie.masterset(South : 3, %0)
      %9 = aie.masterset(North : 5, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 10, %1)
      }
      aie.packet_rules(East : 0) {
        aie.rule(31, 3, %2)
      }
      aie.packet_rules(East : 2) {
        aie.rule(31, 2, %3)
      }
      aie.packet_rules(East : 3) {
        aie.rule(31, 1, %4)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(West : 1, %0)
      %3 = aie.masterset(West : 3, %1)
      aie.packet_rules(East : 0) {
        aie.rule(31, 3, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 2, %0)
      }
    }
    %tile_3_2 = aie.tile(3, 2)
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(South : 0, %1)
      %4 = aie.masterset(South : 2, %2)
      %5 = aie.masterset(West : 0, %0)
      aie.packet_rules(East : 2) {
        aie.rule(31, 5, %2)
      }
      aie.packet_rules(East : 3) {
        aie.rule(31, 4, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %0)
      }
    }
    %tile_4_2 = aie.tile(4, 2)
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(West : 2, %1)
      %3 = aie.masterset(West : 3, %0)
      aie.packet_rules(East : 3) {
        aie.rule(31, 5, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 4, %0)
      }
    }
    %tile_5_2 = aie.tile(5, 2)
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(West : 3, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 5, %0)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<East : 2, South : 2>
      aie.connect<North : 2, South : 3>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 0, %1)
      %3 = aie.masterset(North : 1, %0)
      aie.packet_rules(East : 3) {
        aie.rule(31, 5, %1)
      }
      aie.packet_rules(East : 0) {
        aie.rule(31, 4, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<DMA : 0, East : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(South : 5) {
        aie.rule(31, 10, %0)
      }
    }
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<North : 2, West : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(West : 0, %1)
      %3 = aie.masterset(West : 3, %0)
      aie.packet_rules(East : 1) {
        aie.rule(31, 5, %0)
      }
      aie.packet_rules(East : 3) {
        aie.rule(31, 4, %1)
      }
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<North : 2, South : 2>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<North : 3, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(West : 0, %2)
      %4 = aie.masterset(West : 2, %1)
      %5 = aie.masterset(West : 3, %0)
      aie.packet_rules(East : 3) {
        aie.rule(31, 3, %2)
      }
      aie.packet_rules(East : 1) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }
    }
    %tile_1_3 = aie.tile(1, 3)
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<West : 0, South : 3>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 1, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.amsel<4> (0)
      %5 = aie.amsel<5> (0)
      %6 = aie.amsel<5> (1)
      %7 = aie.masterset(DMA : 0, %5)
      %8 = aie.masterset(DMA : 1, %4)
      %9 = aie.masterset(DMA : 2, %2)
      %10 = aie.masterset(DMA : 3, %3)
      %11 = aie.masterset(DMA : 4, %1)
      %12 = aie.masterset(DMA : 5, %6)
      %13 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 0) {
        aie.rule(31, 5, %6)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 10, %0)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 4, %1)
      }
      aie.packet_rules(North : 1) {
        aie.rule(31, 3, %3)
      }
      aie.packet_rules(North : 0) {
        aie.rule(31, 2, %2)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 1, %4)
      }
      aie.packet_rules(North : 3) {
        aie.rule(31, 0, %5)
      }
    }
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(West : 1, %0)
      %3 = aie.masterset(West : 3, %1)
      aie.packet_rules(East : 2) {
        aie.rule(31, 5, %0)
      }
      aie.packet_rules(East : 3) {
        aie.rule(31, 4, %1)
      }
    }
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(West : 2, %1)
      %3 = aie.masterset(West : 3, %0)
      aie.packet_rules(North : 2) {
        aie.rule(31, 5, %1)
      }
      aie.packet_rules(North : 0) {
        aie.rule(31, 4, %0)
      }
    }
    %mem_tile_3_1 = aie.tile(3, 1)
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 0, %0)
      %3 = aie.masterset(South : 2, %1)
      aie.packet_rules(North : 2) {
        aie.rule(31, 5, %1)
      }
      aie.packet_rules(North : 0) {
        aie.rule(31, 4, %0)
      }
    }
  }
}
