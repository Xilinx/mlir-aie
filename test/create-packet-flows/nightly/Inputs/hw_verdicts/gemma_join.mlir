// HW: PASS
// Join-hazard repro. Core (0,2) consumes A (S2MM 0) and B (S2MM 1) in
// lockstep, one buffer each. A reaches the memtile first (host waits on it
// before sending B), so memtile (0,1) pushes A packet 2 while the core still
// waits for B packet 1. If DMA:0 and DMA:1 masters at (0,2) share an arbiter,
// A packet 2 holds the grant until tlast, which never comes: B cannot pass.
module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %c  = aie.tile(0, 2)

    aie.flow(%s0, DMA : 0, %m0, DMA : 0)
    aie.flow(%s1, DMA : 0, %m1, DMA : 0)
    aie.flow(%c, DMA : 0, %s0, DMA : 0)
    aie.packet_flow(0) { aie.packet_source<%m0, DMA : 0> aie.packet_dest<%c, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m1, DMA : 0> aie.packet_dest<%c, DMA : 1> }

    // ---- memtile (0,1): all of A, then 4 packets (id 0) ----
    %mtA = aie.buffer(%m0) {sym_name = "mtA"} : memref<1024xi32>
    %mtA_prod = aie.lock(%m0, 0) {init = 4 : i32, sym_name = "mtA_prod"}
    %mtA_cons = aie.lock(%m0, 1) {init = 0 : i32, sym_name = "mtA_cons"}
    %md0 = aie.memtile_dma(%m0) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^r0, ^s)
    ^r0:
      aie.use_lock(%mtA_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtA : memref<1024xi32> offset = 0 len = 256)
      aie.use_lock(%mtA_cons, Release, %one)
      aie.next_bd ^r1
    ^r1:
      aie.use_lock(%mtA_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtA : memref<1024xi32> offset = 256 len = 256)
      aie.use_lock(%mtA_cons, Release, %one)
      aie.next_bd ^r2
    ^r2:
      aie.use_lock(%mtA_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtA : memref<1024xi32> offset = 512 len = 256)
      aie.use_lock(%mtA_cons, Release, %one)
      aie.next_bd ^r3
    ^r3:
      aie.use_lock(%mtA_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtA : memref<1024xi32> offset = 768 len = 256)
      aie.use_lock(%mtA_cons, Release, %one)
      aie.next_bd ^r0
    ^s:
      %1 = aie.dma_start(MM2S, 0, ^p0, ^end)
    ^p0:
      aie.use_lock(%mtA_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtA : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%mtA_prod, Release, %one)
      aie.next_bd ^p1
    ^p1:
      aie.use_lock(%mtA_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtA : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%mtA_prod, Release, %one)
      aie.next_bd ^p2
    ^p2:
      aie.use_lock(%mtA_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtA : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%mtA_prod, Release, %one)
      aie.next_bd ^p3
    ^p3:
      aie.use_lock(%mtA_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtA : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%mtA_prod, Release, %one)
      aie.next_bd ^p0
    ^end:
      aie.end
    }

    // ---- memtile (1,1): all of B, then 4 packets (id 1) ----
    %mtB = aie.buffer(%m1) {sym_name = "mtB"} : memref<1024xi32>
    %mtB_prod = aie.lock(%m1, 0) {init = 4 : i32, sym_name = "mtB_prod"}
    %mtB_cons = aie.lock(%m1, 1) {init = 0 : i32, sym_name = "mtB_cons"}
    %md1 = aie.memtile_dma(%m1) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^r0, ^s)
    ^r0:
      aie.use_lock(%mtB_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtB : memref<1024xi32> offset = 0 len = 256)
      aie.use_lock(%mtB_cons, Release, %one)
      aie.next_bd ^r1
    ^r1:
      aie.use_lock(%mtB_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtB : memref<1024xi32> offset = 256 len = 256)
      aie.use_lock(%mtB_cons, Release, %one)
      aie.next_bd ^r2
    ^r2:
      aie.use_lock(%mtB_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtB : memref<1024xi32> offset = 512 len = 256)
      aie.use_lock(%mtB_cons, Release, %one)
      aie.next_bd ^r3
    ^r3:
      aie.use_lock(%mtB_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtB : memref<1024xi32> offset = 768 len = 256)
      aie.use_lock(%mtB_cons, Release, %one)
      aie.next_bd ^r0
    ^s:
      %1 = aie.dma_start(MM2S, 0, ^p0, ^end)
    ^p0:
      aie.use_lock(%mtB_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtB : memref<1024xi32> offset = 0 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%mtB_prod, Release, %one)
      aie.next_bd ^p1
    ^p1:
      aie.use_lock(%mtB_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtB : memref<1024xi32> offset = 256 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%mtB_prod, Release, %one)
      aie.next_bd ^p2
    ^p2:
      aie.use_lock(%mtB_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtB : memref<1024xi32> offset = 512 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%mtB_prod, Release, %one)
      aie.next_bd ^p3
    ^p3:
      aie.use_lock(%mtB_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%mtB : memref<1024xi32> offset = 768 len = 256) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.use_lock(%mtB_prod, Release, %one)
      aie.next_bd ^p0
    ^end:
      aie.end
    }

    // ---- core (0,2): O = A + B, one A and one B per iteration ----
    %bA = aie.buffer(%c) {sym_name = "bA"} : memref<256xi32>
    %bB = aie.buffer(%c) {sym_name = "bB"} : memref<256xi32>
    %bO = aie.buffer(%c) {sym_name = "bO"} : memref<256xi32>
    %a_prod = aie.lock(%c, 0) {init = 1 : i32, sym_name = "a_prod"}
    %a_cons = aie.lock(%c, 1) {init = 0 : i32, sym_name = "a_cons"}
    %b_prod = aie.lock(%c, 2) {init = 1 : i32, sym_name = "b_prod"}
    %b_cons = aie.lock(%c, 3) {init = 0 : i32, sym_name = "b_cons"}
    %o_prod = aie.lock(%c, 4) {init = 1 : i32, sym_name = "o_prod"}
    %o_cons = aie.lock(%c, 5) {init = 0 : i32, sym_name = "o_cons"}

    %core = aie.core(%c) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c256 = arith.constant 256 : index
      scf.for %it = %c0 to %c4 step %c1 {
        aie.use_lock(%a_cons, AcquireGreaterEqual, %one)
        aie.use_lock(%b_cons, AcquireGreaterEqual, %one)
        aie.use_lock(%o_prod, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %c256 step %c1 {
          %x = memref.load %bA[%i] : memref<256xi32>
          %y = memref.load %bB[%i] : memref<256xi32>
          %z = arith.addi %x, %y : i32
          memref.store %z, %bO[%i] : memref<256xi32>
        }
        aie.use_lock(%a_prod, Release, %one)
        aie.use_lock(%b_prod, Release, %one)
        aie.use_lock(%o_cons, Release, %one)
      }
      aie.end
    }

    %mem = aie.mem(%c) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^a, ^s1)
    ^a:
      aie.use_lock(%a_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%bA : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%a_cons, Release, %one)
      aie.next_bd ^a
    ^s1:
      %1 = aie.dma_start(S2MM, 1, ^b, ^s2)
    ^b:
      aie.use_lock(%b_prod, AcquireGreaterEqual, %one)
      aie.dma_bd(%bB : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%b_cons, Release, %one)
      aie.next_bd ^b
    ^s2:
      %2 = aie.dma_start(MM2S, 0, ^o, ^end)
    ^o:
      aie.use_lock(%o_cons, AcquireGreaterEqual, %one)
      aie.dma_bd(%bO : memref<256xi32> offset = 0 len = 256)
      aie.use_lock(%o_prod, Release, %one)
      aie.next_bd ^o
    ^end:
      aie.end
    }

    aie.shim_dma_allocation @inA (%s0, MM2S, 0)
    aie.shim_dma_allocation @inB (%s1, MM2S, 0)
    aie.shim_dma_allocation @out (%s0, S2MM, 0)

    aie.runtime_sequence(%a: memref<1024xi32>, %b: memref<1024xi32>, %o: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd (%a[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA, issue_token = true} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd (%o[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, metadata = @out, issue_token = true} : memref<1024xi32>
      aiex.npu.dma_wait { symbol = @inA }
      aiex.npu.dma_memcpy_nd (%b[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 2 : i64, metadata = @inB} : memref<1024xi32>
      aiex.npu.dma_wait { symbol = @out }
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
        aie.rule(31, 0, %0)
      }
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, DMA : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 0, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %0)
      %3 = aie.masterset(DMA : 1, %1)
      aie.packet_rules(East : 2) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(West : 2, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %0)
      }
    }
  }
}
