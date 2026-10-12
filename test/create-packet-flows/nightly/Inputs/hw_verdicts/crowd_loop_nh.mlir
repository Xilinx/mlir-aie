// HW: PASS
// hops: off
// Looping chains at (0,1): 8 masters, unknown volumes, still routable.
module {
  aie.device(npu2) {
    %t0_0 = aie.tile(0, 0)
    %t1_0 = aie.tile(1, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    %t0_3 = aie.tile(0, 3)
    aie.packet_flow(20) { aie.packet_source<%t0_0, DMA : 0> aie.packet_dest<%t0_1, DMA : 0> }
    aie.packet_flow(21) { aie.packet_source<%t0_0, DMA : 1> aie.packet_dest<%t0_1, DMA : 1> }
    aie.packet_flow(22) { aie.packet_source<%t1_0, DMA : 0> aie.packet_dest<%t0_1, DMA : 2> }
    aie.packet_flow(23) { aie.packet_source<%t1_0, DMA : 1> aie.packet_dest<%t0_1, DMA : 3> }
    aie.packet_flow(0) { aie.packet_source<%t0_1, DMA : 0> aie.packet_dest<%t0_2, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t0_1, DMA : 1> aie.packet_dest<%t0_2, DMA : 1> }
    aie.flow(%t0_2, DMA : 0, %t0_0, DMA : 0)
    aie.packet_flow(2) { aie.packet_source<%t0_1, DMA : 2> aie.packet_dest<%t0_3, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%t0_1, DMA : 3> aie.packet_dest<%t0_3, DMA : 1> }
    aie.flow(%t0_3, DMA : 0, %t1_0, DMA : 0)
    %mt0 = aie.buffer(%t0_1) {sym_name = "mt0"} : memref<1024xi32>
    %mt0_p = aie.lock(%t0_1, 0) {init = 4 : i32, sym_name = "mt0_p"}
    %mt0_c = aie.lock(%t0_1, 1) {init = 0 : i32, sym_name = "mt0_c"}
    %mt1 = aie.buffer(%t0_1) {sym_name = "mt1"} : memref<1024xi32>
    %mt1_p = aie.lock(%t0_1, 2) {init = 4 : i32, sym_name = "mt1_p"}
    %mt1_c = aie.lock(%t0_1, 3) {init = 0 : i32, sym_name = "mt1_c"}
    %mt2 = aie.buffer(%t0_1) {sym_name = "mt2"} : memref<1024xi32>
    %mt2_p = aie.lock(%t0_1, 4) {init = 4 : i32, sym_name = "mt2_p"}
    %mt2_c = aie.lock(%t0_1, 5) {init = 0 : i32, sym_name = "mt2_c"}
    %mt3 = aie.buffer(%t0_1) {sym_name = "mt3"} : memref<1024xi32>
    %mt3_p = aie.lock(%t0_1, 6) {init = 4 : i32, sym_name = "mt3_p"}
    %mt3_c = aie.lock(%t0_1, 7) {init = 0 : i32, sym_name = "mt3_c"}
    %i02_0 = aie.buffer(%t0_2) {sym_name = "i02_0"} : memref<256xi32>
    %i02_0_p = aie.lock(%t0_2, 0) {init = 1 : i32, sym_name = "i02_0_p"}
    %i02_0_c = aie.lock(%t0_2, 1) {init = 0 : i32, sym_name = "i02_0_c"}
    %i02_1 = aie.buffer(%t0_2) {sym_name = "i02_1"} : memref<256xi32>
    %i02_1_p = aie.lock(%t0_2, 2) {init = 1 : i32, sym_name = "i02_1_p"}
    %i02_1_c = aie.lock(%t0_2, 3) {init = 0 : i32, sym_name = "i02_1_c"}
    %o02_0 = aie.buffer(%t0_2) {sym_name = "o02_0"} : memref<256xi32>
    %o02_0_p = aie.lock(%t0_2, 4) {init = 1 : i32, sym_name = "o02_0_p"}
    %o02_0_c = aie.lock(%t0_2, 5) {init = 0 : i32, sym_name = "o02_0_c"}
    %i03_0 = aie.buffer(%t0_3) {sym_name = "i03_0"} : memref<256xi32>
    %i03_0_p = aie.lock(%t0_3, 0) {init = 1 : i32, sym_name = "i03_0_p"}
    %i03_0_c = aie.lock(%t0_3, 1) {init = 0 : i32, sym_name = "i03_0_c"}
    %i03_1 = aie.buffer(%t0_3) {sym_name = "i03_1"} : memref<256xi32>
    %i03_1_p = aie.lock(%t0_3, 2) {init = 1 : i32, sym_name = "i03_1_p"}
    %i03_1_c = aie.lock(%t0_3, 3) {init = 0 : i32, sym_name = "i03_1_c"}
    %o03_0 = aie.buffer(%t0_3) {sym_name = "o03_0"} : memref<256xi32>
    %o03_0_p = aie.lock(%t0_3, 4) {init = 1 : i32, sym_name = "o03_0_p"}
    %o03_0_c = aie.lock(%t0_3, 5) {init = 0 : i32, sym_name = "o03_0_c"}
    %core_t0_2 = aie.core(%t0_2) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
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
    %core_t0_3 = aie.core(%t0_3) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cb = arith.constant 256 : index
      %k = arith.constant 0 : i32
      scf.for %it = %c0 to %cn step %c1 {
        aie.use_lock(%i03_0_c, AcquireGreaterEqual, %one)
        aie.use_lock(%i03_1_c, AcquireGreaterEqual, %one)
        aie.use_lock(%o03_0_p, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %cb step %c1 {
          %a0 = arith.addi %i, %c0 : index
          %x0 = memref.load %i03_0[%a0] : memref<256xi32>
          %s0 = arith.addi %k, %x0 : i32
          %a1 = arith.addi %i, %c0 : index
          %x1 = memref.load %i03_1[%a1] : memref<256xi32>
          %s1 = arith.addi %s0, %x1 : i32
          %ao = arith.addi %i, %c0 : index
          memref.store %s1, %o03_0[%ao] : memref<256xi32>
        }
        aie.use_lock(%i03_0_p, Release, %one)
        aie.use_lock(%i03_1_p, Release, %one)
        aie.use_lock(%o03_0_c, Release, %one)
      }
      aie.end
    }
    %dma_t0_1 = aie.memtile_dma(%t0_1) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^c0b1
      ^c0b1:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^c0b2
      ^c0b2:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^c0b3
      ^c0b3:
        aie.use_lock(%mt0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(%mt0_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(MM2S, 0, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^c1b1
      ^c1b1:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^c1b2
      ^c1b2:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^c1b3
      ^c1b3:
        aie.use_lock(%mt0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt0 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%mt0_p, Release, %one)
        aie.next_bd ^c1b0
      ^s2:
      %d2 = aie.dma_start(S2MM, 1, ^c2b0, ^s3)
      ^c2b0:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^c2b1
      ^c2b1:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^c2b2
      ^c2b2:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^c2b3
      ^c2b3:
        aie.use_lock(%mt1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(%mt1_c, Release, %one)
        aie.next_bd ^c2b0
      ^s3:
      %d3 = aie.dma_start(MM2S, 1, ^c3b0, ^s4)
      ^c3b0:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^c3b1
      ^c3b1:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^c3b2
      ^c3b2:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^c3b3
      ^c3b3:
        aie.use_lock(%mt1_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt1 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%mt1_p, Release, %one)
        aie.next_bd ^c3b0
      ^s4:
      %d4 = aie.dma_start(S2MM, 2, ^c4b0, ^s5)
      ^c4b0:
        aie.use_lock(%mt2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(%mt2_c, Release, %one)
        aie.next_bd ^c4b1
      ^c4b1:
        aie.use_lock(%mt2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(%mt2_c, Release, %one)
        aie.next_bd ^c4b2
      ^c4b2:
        aie.use_lock(%mt2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(%mt2_c, Release, %one)
        aie.next_bd ^c4b3
      ^c4b3:
        aie.use_lock(%mt2_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(%mt2_c, Release, %one)
        aie.next_bd ^c4b0
      ^s5:
      %d5 = aie.dma_start(MM2S, 2, ^c5b0, ^s6)
      ^c5b0:
        aie.use_lock(%mt2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%mt2_p, Release, %one)
        aie.next_bd ^c5b1
      ^c5b1:
        aie.use_lock(%mt2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%mt2_p, Release, %one)
        aie.next_bd ^c5b2
      ^c5b2:
        aie.use_lock(%mt2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%mt2_p, Release, %one)
        aie.next_bd ^c5b3
      ^c5b3:
        aie.use_lock(%mt2_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt2 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%mt2_p, Release, %one)
        aie.next_bd ^c5b0
      ^s6:
      %d6 = aie.dma_start(S2MM, 3, ^c6b0, ^s7)
      ^c6b0:
        aie.use_lock(%mt3_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt3 : memref<1024xi32> offset = 0 len = 256)
        aie.use_lock(%mt3_c, Release, %one)
        aie.next_bd ^c6b1
      ^c6b1:
        aie.use_lock(%mt3_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt3 : memref<1024xi32> offset = 256 len = 256)
        aie.use_lock(%mt3_c, Release, %one)
        aie.next_bd ^c6b2
      ^c6b2:
        aie.use_lock(%mt3_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt3 : memref<1024xi32> offset = 512 len = 256)
        aie.use_lock(%mt3_c, Release, %one)
        aie.next_bd ^c6b3
      ^c6b3:
        aie.use_lock(%mt3_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt3 : memref<1024xi32> offset = 768 len = 256)
        aie.use_lock(%mt3_c, Release, %one)
        aie.next_bd ^c6b0
      ^s7:
      %d7 = aie.dma_start(MM2S, 3, ^c7b0, ^end)
      ^c7b0:
        aie.use_lock(%mt3_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt3 : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
        aie.use_lock(%mt3_p, Release, %one)
        aie.next_bd ^c7b1
      ^c7b1:
        aie.use_lock(%mt3_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt3 : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
        aie.use_lock(%mt3_p, Release, %one)
        aie.next_bd ^c7b2
      ^c7b2:
        aie.use_lock(%mt3_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt3 : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
        aie.use_lock(%mt3_p, Release, %one)
        aie.next_bd ^c7b3
      ^c7b3:
        aie.use_lock(%mt3_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%mt3 : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
        aie.use_lock(%mt3_p, Release, %one)
        aie.next_bd ^c7b0
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
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(S2MM, 1, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%i02_1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i02_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i02_1_c, Release, %one)
        aie.next_bd ^c1b0
      ^s2:
      %d2 = aie.dma_start(MM2S, 0, ^c2b0, ^end)
      ^c2b0:
        aie.use_lock(%o02_0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%o02_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%o02_0_p, Release, %one)
        aie.next_bd ^c2b0
      ^end:
        aie.end
    }
    %dma_t0_3 = aie.mem(%t0_3) {
      %one = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^c0b0, ^s1)
      ^c0b0:
        aie.use_lock(%i03_0_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i03_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i03_0_c, Release, %one)
        aie.next_bd ^c0b0
      ^s1:
      %d1 = aie.dma_start(S2MM, 1, ^c1b0, ^s2)
      ^c1b0:
        aie.use_lock(%i03_1_p, AcquireGreaterEqual, %one)
        aie.dma_bd(%i03_1 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%i03_1_c, Release, %one)
        aie.next_bd ^c1b0
      ^s2:
      %d2 = aie.dma_start(MM2S, 0, ^c2b0, ^end)
      ^c2b0:
        aie.use_lock(%o03_0_c, AcquireGreaterEqual, %one)
        aie.dma_bd(%o03_0 : memref<256xi32> offset = 0 len = 256)
        aie.use_lock(%o03_0_p, Release, %one)
        aie.next_bd ^c2b0
      ^end:
        aie.end
    }
    aie.shim_dma_allocation @in0(%t0_0, MM2S, 0)
    aie.shim_dma_allocation @in1(%t0_0, MM2S, 1)
    aie.shim_dma_allocation @in2(%t1_0, MM2S, 0)
    aie.shim_dma_allocation @in3(%t1_0, MM2S, 1)
    aie.shim_dma_allocation @out0(%t0_0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t1_0, S2MM, 0)
    aie.runtime_sequence(%in: memref<4096xi32>, %out: memref<2048xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1], packet = <pkt_id = 20, pkt_type = 0>) {id = 0 : i64, issue_token = true, metadata = @in0} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 2048][1, 1, 1, 1024][0, 0, 0, 1], packet = <pkt_id = 22, pkt_type = 0>) {id = 0 : i64, issue_token = true, metadata = @in2} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out0} : memref<2048xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 1024][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out1} : memref<2048xi32>
      aiex.npu.dma_wait {symbol = @in0}
      aiex.npu.dma_wait {symbol = @in2}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 1024][1, 1, 1, 1024][0, 0, 0, 1], packet = <pkt_id = 21, pkt_type = 0>) {id = 2 : i64, issue_token = true, metadata = @in1} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 3072][1, 1, 1, 1024][0, 0, 0, 1], packet = <pkt_id = 23, pkt_type = 0>) {id = 2 : i64, issue_token = true, metadata = @in3} : memref<4096xi32>
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, East : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.masterset(North : 2, %2)
      %5 = aie.masterset(North : 3, %0)
      %6 = aie.masterset(North : 4, %3)
      %7 = aie.masterset(North : 5, %1)
      aie.packet_rules(East : 1) {
        aie.rule(31, 23, %2)
      }
      aie.packet_rules(East : 2) {
        aie.rule(31, 22, %3)
      }
      aie.packet_rules(South : 7) {
        aie.rule(31, 21, %1)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 20, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<North : 0, South : 0>
      aie.connect<North : 1, South : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.amsel<4> (0)
      %5 = aie.amsel<5> (0)
      %6 = aie.amsel<4> (1)
      %7 = aie.amsel<5> (1)
      %8 = aie.masterset(DMA : 0, %5)
      %9 = aie.masterset(DMA : 1, %7)
      %10 = aie.masterset(DMA : 2, %6)
      %11 = aie.masterset(DMA : 3, %4)
      %12 = aie.masterset(North : 0, %2)
      %13 = aie.masterset(North : 1, %0)
      %14 = aie.masterset(North : 4, %3)
      %15 = aie.masterset(North : 5, %1)
      aie.packet_rules(DMA : 3) {
        aie.rule(31, 3, %3)
      }
      aie.packet_rules(DMA : 2) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 1, %2)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 23, %4)
      }
      aie.packet_rules(South : 4) {
        aie.rule(31, 22, %6)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 21, %7)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 20, %5)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, South : 0>
      aie.connect<North : 0, South : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.masterset(DMA : 0, %0)
      %5 = aie.masterset(DMA : 1, %2)
      %6 = aie.masterset(North : 1, %1)
      %7 = aie.masterset(North : 3, %3)
      aie.packet_rules(South : 4) {
        aie.rule(31, 3, %3)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(South : 0) {
        aie.rule(31, 1, %2)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %0)
      }
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %0)
      %3 = aie.masterset(DMA : 1, %1)
      aie.packet_rules(South : 3) {
        aie.rule(31, 3, %1)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 2, %0)
      }
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<West : 1, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(West : 1, %1)
      %3 = aie.masterset(West : 2, %0)
      aie.packet_rules(South : 7) {
        aie.rule(31, 23, %1)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 22, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
  }
}
