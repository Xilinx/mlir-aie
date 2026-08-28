//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Spike: can a compute tile's own DMA source a control-packet flow into
// another tile's TileControl port, or is the shim the only legal source?
//
// RESOLVED 2026-08-24, hardware-verified (Strix, 4 consecutive runs): yes.
//
// This is a minimal diff from the known-good add_one_ctrl_packet_4_cores
// test: three of its four control-packet flows are untouched (shim ->
// TileControl, releasing tile_0_2/0_3/0_4's locks exactly as before). The
// fourth (pkt_id=8, into tile_0_5) is re-sourced from a new compute tile,
// (1, 2), carrying the *identical* payload words tile_0_5 already receives
// from the host upstream (same lock addresses, same parity math -- see
// test.cpp for the host-side version of the same computation), written into
// tile_1_2's own local memory by its own core instead of arriving from DDR
// via the shim. tile_0_5 comes up and produces the correct add-one output,
// same as every other core in the design.
//
// Two real, general bugs stood between "architecturally plausible" and
// "works", neither of them about TileControl specifically. Both will bite
// anyone building on this mechanism, so they are recorded here rather than
// just fixed silently:
//
// 1. A compute tile's DMA channel gated by a lock nothing else ever touches
//    (an "always start at 1, never replenished" or "self-consuming" trigger
//    with no producer/consumer pairing tied to a real core action) never
//    actually starts the hardware queue -- the run times out waiting on a
//    transfer that never happens, with no error. This looked exactly like a
//    TileControl/payload-format bug for several iterations; it was not. Fix:
//    the same idiom every other tile in this file uses -- a real producer
//    lock, a core action that releases a real consumer lock, and the DMA
//    gated on that. See test/npu-xrt/minimal_tile_send_spike for the
//    isolated 2-tile proof this alone fixes a plain (non-TileControl)
//    tile-to-shim send.
//
// 2. Even with (1) fixed, combining both lock-release control packets
//    (lock0's and lock2's, 2 words each) into a single 4-word DMA burst does
//    not work. The host-sourced baseline never does this: read
//    add_one_ctrl_packet_4_cores/aie.mlir's runtime_sequence closely and
//    every dma_memcpy_nd transfer carrying a control packet is exactly 2
//    words (one header, one data word), individually tagged
//    packet=<pkt_id, pkt_type=1>. Two control packets means two separate
//    DMA bursts, never one burst carrying both. The fix here chains two
//    aie.dma_bd entries (mem_1_2, offset 0 len 2, then offset 2 len 2),
//    each independently packet-tagged, matching that shape exactly.
//
// Between these two, "why doesn't a payload format that looks correct byte-
// for-byte actually work" had two different answers, and neither was about
// the wire-level header encoding this file spent the most effort on
// initially -- it was about *how many DMA bursts* carry it and *what makes
// the DMA queue start running at all*.
//
// 3. Found later, building iron.overlay's `ProgramMemorySlot(source=...)` on
//    top of this spike (see python/iron/overlay/slot.py's
//    _load_tile_sourced): a single packet-tagged `aie.dma_bd` with a
//    `#aie.bd_iteration<size, stride, ...>` attribute -- the obvious way to
//    cover many chunks with one hardware descriptor instead of one per
//    chunk -- corrupts the packet's embedded address on every execution but
//    the first (hardware-verified, Strix: a 2-chunk write with iteration
//    hangs the target core forever; the identical 2 chunks as two separate,
//    non-iterated `aie.dma_bd` entries, chained the same way as (2) above,
//    lands correctly every time). Root cause not isolated further than
//    that -- the fix is simply: one explicit, non-iterated `aie.dma_bd` per
//    control-packet chunk, which caps how many chunks a single source
//    tile's whole BD table (16 entries on Strix,
//    `AIETargetModel::getNumBDs()`, shared across every DMA channel on the
//    tile) can carry in one static configuration -- there is no
//    iteration-based way around that budget today.
module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    // Pure bystander: not on any of the four cores' control or data path.
    %tile_1_2 = aie.tile(1, 2)

    %input_0_2_lock0 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "input_0_2_lock0"}
    %input_0_2_lock2 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "input_0_2_lock2"}
    %output_0_2_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "output_0_2_lock4"}
    %output_0_2_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "output_0_2_lock5"}

    %input_0_2_buffer = aie.buffer(%tile_0_2) {sym_name = "input_0_2_buffer"} : memref<8xi32>
    %output_0_2_buffer = aie.buffer(%tile_0_2) {sym_name = "output_0_2_buffer"} : memref<8xi32>

    %input_0_3_lock0 = aie.lock(%tile_0_3, 0) {init = 0 : i32, sym_name = "input_0_3_lock0"}
    %input_0_3_lock2 = aie.lock(%tile_0_3, 2) {init = 0 : i32, sym_name = "input_0_3_lock2"}
    %output_0_3_lock4 = aie.lock(%tile_0_3, 4) {init = 0 : i32, sym_name = "output_0_3_lock4"}
    %output_0_3_lock5 = aie.lock(%tile_0_3, 5) {init = 1 : i32, sym_name = "output_0_3_lock5"}

    %input_0_3_buffer = aie.buffer(%tile_0_3) {sym_name = "input_0_3_buffer"} : memref<8xi32>
    %output_0_3_buffer = aie.buffer(%tile_0_3) {sym_name = "output_0_3_buffer"} : memref<8xi32>

    %input_0_4_lock0 = aie.lock(%tile_0_4, 0) {init = 0 : i32, sym_name = "input_0_4_lock0"}
    %input_0_4_lock2 = aie.lock(%tile_0_4, 2) {init = 0 : i32, sym_name = "input_0_4_lock2"}
    %output_0_4_lock4 = aie.lock(%tile_0_4, 4) {init = 0 : i32, sym_name = "output_0_4_lock4"}
    %output_0_4_lock5 = aie.lock(%tile_0_4, 5) {init = 1 : i32, sym_name = "output_0_4_lock5"}

    %input_0_4_buffer = aie.buffer(%tile_0_4) {sym_name = "input_0_4_buffer"} : memref<8xi32>
    %output_0_4_buffer = aie.buffer(%tile_0_4) {sym_name = "output_0_4_buffer"} : memref<8xi32>

    %input_0_5_lock0 = aie.lock(%tile_0_5, 0) {init = 0 : i32, sym_name = "input_0_5_lock0"}
    %input_0_5_lock2 = aie.lock(%tile_0_5, 2) {init = 0 : i32, sym_name = "input_0_5_lock2"}
    %output_0_5_lock4 = aie.lock(%tile_0_5, 4) {init = 0 : i32, sym_name = "output_0_5_lock4"}
    %output_0_5_lock5 = aie.lock(%tile_0_5, 5) {init = 1 : i32, sym_name = "output_0_5_lock5"}

    %input_0_5_buffer = aie.buffer(%tile_0_5) {sym_name = "input_0_5_buffer"} : memref<8xi32>
    %output_0_5_buffer = aie.buffer(%tile_0_5) {sym_name = "output_0_5_buffer"} : memref<8xi32>

    // Real producer/consumer locks, same idiom as every other tile in this
    // file (and as minimal_tile_send_spike, once fixed): the core writes and
    // releases; the DMA acquires and sends. A self-looping "always ready"
    // lock with no core action driving it never actually starts the DMA
    // channel's hardware queue -- see minimal_tile_send_spike's history for
    // the isolated proof.
    %pkt8_payload = aie.buffer(%tile_1_2) {sym_name = "pkt8_payload"} : memref<4xi32>
    %pkt8_prod_lock = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "pkt8_prod_lock"}
    %pkt8_cons_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "pkt8_cons_lock"}
    // Hand-off between the two chained BDs below: each block needs its own
    // paired acquire/release, so the first BD's block releases this instead
    // of pkt8_prod_lock directly, and the second BD's block picks it up.
    %pkt8_mid_lock = aie.lock(%tile_1_2, 2) {init = 0 : i32, sym_name = "pkt8_mid_lock"}

    aie.packet_flow(0x5) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    }
    aie.packet_flow(0x6) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_3, TileControl : 0>
    }
    aie.packet_flow(0x7) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_4, TileControl : 0>
    }
    // The one change from upstream add_one_ctrl_packet_4_cores: sourced from
    // a compute tile's own DMA, not the shim. keep_pkt_header/priority_route
    // mirror AIEGenerateColumnControlOverlay's own construction of
    // shim->TileControl flows.
    aie.packet_flow(0x8) {
      aie.packet_source<%tile_1_2, DMA : 0>
      aie.packet_dest<%tile_0_5, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 1)
    aie.flow(%tile_0_3, DMA : 0, %tile_1_0, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %tile_1_0, DMA : 1)
    aie.flow(%tile_0_5, DMA : 0, %tile_2_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %arg1_i32 = arith.index_cast %arg1 : index to i32
        %1 = arith.addi %arg1_i32, %c3_i32 : i32
        memref.store %1, %input_0_2_buffer[%arg1] : memref<8xi32>
      }
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%input_0_2_lock0, AcquireGreaterEqual, %c1_ul1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_2_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%input_0_2_lock0, AcquireGreaterEqual, %c1_ul2)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_2_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%input_0_2_lock2, AcquireGreaterEqual, %c1_ul3)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_2_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%input_0_2_lock2, AcquireGreaterEqual, %c1_ul4)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_2_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%output_0_2_lock5, AcquireGreaterEqual, %c1_ul5)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_0_2_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%output_0_2_lock4, Release, %c1_ul6)
      }
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(%output_0_2_lock4, AcquireGreaterEqual, %c1_ul7)
      aie.dma_bd(%output_0_2_buffer : memref<8xi32> offset = 0 len = 8)
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(%output_0_2_lock5, Release, %c1_ul8)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %arg1_i32 = arith.index_cast %arg1 : index to i32
        %1 = arith.addi %arg1_i32, %c3_i32 : i32
        memref.store %1, %input_0_3_buffer[%arg1] : memref<8xi32>
      }
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(%input_0_3_lock0, AcquireGreaterEqual, %c1_ul9)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_3_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%input_0_3_lock0, AcquireGreaterEqual, %c1_ul10)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_3_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(%input_0_3_lock2, AcquireGreaterEqual, %c1_ul11)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_3_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(%input_0_3_lock2, AcquireGreaterEqual, %c1_ul12)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_3_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(%output_0_3_lock5, AcquireGreaterEqual, %c1_ul13)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_0_3_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(%output_0_3_lock4, Release, %c1_ul14)
      }
      aie.end
    }

    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_ul15 = arith.constant 1 : i32
      aie.use_lock(%output_0_3_lock4, AcquireGreaterEqual, %c1_ul15)
      aie.dma_bd(%output_0_3_buffer : memref<8xi32> offset = 0 len = 8)
      %c1_ul16 = arith.constant 1 : i32
      aie.use_lock(%output_0_3_lock5, Release, %c1_ul16)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %arg1_i32 = arith.index_cast %arg1 : index to i32
        %1 = arith.addi %arg1_i32, %c3_i32 : i32
        memref.store %1, %input_0_4_buffer[%arg1] : memref<8xi32>
      }
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c1_ul17 = arith.constant 1 : i32
        aie.use_lock(%input_0_4_lock0, AcquireGreaterEqual, %c1_ul17)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_4_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul18 = arith.constant 1 : i32
        aie.use_lock(%input_0_4_lock0, AcquireGreaterEqual, %c1_ul18)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_4_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul19 = arith.constant 1 : i32
        aie.use_lock(%input_0_4_lock2, AcquireGreaterEqual, %c1_ul19)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_4_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul20 = arith.constant 1 : i32
        aie.use_lock(%input_0_4_lock2, AcquireGreaterEqual, %c1_ul20)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_4_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul21 = arith.constant 1 : i32
        aie.use_lock(%output_0_4_lock5, AcquireGreaterEqual, %c1_ul21)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_0_4_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul22 = arith.constant 1 : i32
        aie.use_lock(%output_0_4_lock4, Release, %c1_ul22)
      }
      aie.end
    }

    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_ul23 = arith.constant 1 : i32
      aie.use_lock(%output_0_4_lock4, AcquireGreaterEqual, %c1_ul23)
      aie.dma_bd(%output_0_4_buffer : memref<8xi32> offset = 0 len = 8)
      %c1_ul24 = arith.constant 1 : i32
      aie.use_lock(%output_0_4_lock5, Release, %c1_ul24)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %arg1_i32 = arith.index_cast %arg1 : index to i32
        %1 = arith.addi %arg1_i32, %c3_i32 : i32
        memref.store %1, %input_0_5_buffer[%arg1] : memref<8xi32>
      }
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c1_ul25 = arith.constant 1 : i32
        aie.use_lock(%input_0_5_lock0, AcquireGreaterEqual, %c1_ul25)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_5_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul26 = arith.constant 1 : i32
        aie.use_lock(%input_0_5_lock0, AcquireGreaterEqual, %c1_ul26)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_5_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul27 = arith.constant 1 : i32
        aie.use_lock(%input_0_5_lock2, AcquireGreaterEqual, %c1_ul27)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_5_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul28 = arith.constant 1 : i32
        aie.use_lock(%input_0_5_lock2, AcquireGreaterEqual, %c1_ul28)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_5_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul29 = arith.constant 1 : i32
        aie.use_lock(%output_0_5_lock5, AcquireGreaterEqual, %c1_ul29)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_0_5_buffer[%arg1] : memref<8xi32>
        }
        %c1_ul30 = arith.constant 1 : i32
        aie.use_lock(%output_0_5_lock4, Release, %c1_ul30)
      }
      aie.end
    }

    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_ul31 = arith.constant 1 : i32
      aie.use_lock(%output_0_5_lock4, AcquireGreaterEqual, %c1_ul31)
      aie.dma_bd(%output_0_5_buffer : memref<8xi32> offset = 0 len = 8)
      %c1_ul32 = arith.constant 1 : i32
      aie.use_lock(%output_0_5_lock5, Release, %c1_ul32)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    // tile_1_2's core: writes the same lock-set control-packet payload
    // tile_0_5 already receives from the host in upstream
    // add_one_ctrl_packet_4_cores (address 0x1F000 is lock0's value
    // register, 0x1F020 is lock2's, both set to 2, stream_id=0, opcode=0,
    // beats=0, even-parity bit31 -- see test.cpp for the identical
    // computation), then releases the real consumer lock that gates the
    // send below. This is the fix: a core action drives the release, not a
    // self-looping lock nothing else ever touches.
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %h0 = arith.constant 0x0001F000 : i32
      %h1 = arith.constant 0x8001F020 : i32
      %d = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%pkt8_prod_lock, AcquireGreaterEqual, %c1_i32)
      memref.store %h0, %pkt8_payload[%c0] : memref<4xi32>
      memref.store %d, %pkt8_payload[%c1] : memref<4xi32>
      memref.store %h1, %pkt8_payload[%c2] : memref<4xi32>
      memref.store %d, %pkt8_payload[%c3] : memref<4xi32>
      aie.use_lock(%pkt8_cons_lock, Release, %c1_i32)
      aie.end
    }

    // The fix that actually mattered: the host sends each control packet
    // (one lock's header+data, 2 words) as its own separate DMA burst --
    // see add_one_ctrl_packet_4_cores/aie.mlir's runtime_sequence, three
    // distinct dma_memcpy_nd calls of size 2, each individually tagged
    // packet=<pkt_id, pkt_type=1>, never four words combined into one
    // transfer. This mem block now matches that shape: two chained BDs, each
    // 2 words, each carrying its own BD-native packet tag, instead of one BD
    // sending all 4 words as a single burst.
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%pkt8_cons_lock, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%pkt8_payload : memref<4xi32> offset = 0 len = 2) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 8>}
      aie.use_lock(%pkt8_mid_lock, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb2:
      aie.use_lock(%pkt8_mid_lock, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%pkt8_payload : memref<4xi32> offset = 2 len = 2) {packet = #aie.packet_info<pkt_type = 1, pkt_id = 8>}
      aie.use_lock(%pkt8_prod_lock, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb3:
      aie.end
    }

    aie.shim_dma_allocation @ctrlin0 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @out1 (%tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @out2 (%tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @out3 (%tile_2_0, S2MM, 0)

    // Same runtime sequence as add_one_ctrl_packet_4_cores, minus the
    // ctrlin1 writes for tile_0_5's payload -- that payload no longer comes
    // from the host at all.
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: memref<32xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c4_i64 = arith.constant 4 : i64
      %c6_i64 = arith.constant 6 : i64
      %c8_i64 = arith.constant 8 : i64
      %c10_i64 = arith.constant 10 : i64
      %c16_i64 = arith.constant 16 : i64
      %c24_i64 = arith.constant 24 : i64

      // start reading output
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @ctrl0} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 2 : i64, issue_token = true, metadata = @out0} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c8_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 3 : i64, issue_token = true, metadata = @out1} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c16_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 4 : i64, issue_token = true, metadata = @out2} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c24_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 5 : i64, issue_token = true, metadata = @out3} : memref<32xi32>

      // write bd0
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 5, pkt_type = 1>) {id = 6 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c4_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 6, pkt_type = 1>) {id = 7 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c8_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 7, pkt_type = 1>) {id = 8 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      %cst_npu_0 = arith.constant 0 : i32
      %cst_npu_1 = arith.constant 0 : i32
      %cst_npu_2 = arith.constant 1 : i32
      %cst_npu_3 = arith.constant 0 : i32
      %cst_npu_4 = arith.constant 1 : i32
      %cst_npu_5 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_0, %cst_npu_1, %cst_npu_2, %cst_npu_3, %cst_npu_4, %cst_npu_5) : i32, i32, i32, i32, i32, i32

      // patch bd0 address for packet 1, push to mm2s_0_task_queue, wait
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c2_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 5, pkt_type = 1>) {id = 6 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c6_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 6, pkt_type = 1>) {id = 7 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c10_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 7, pkt_type = 1>) {id = 8 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      %cst_npu_6 = arith.constant 0 : i32
      %cst_npu_7 = arith.constant 0 : i32
      %cst_npu_8 = arith.constant 1 : i32
      %cst_npu_9 = arith.constant 0 : i32
      %cst_npu_10 = arith.constant 1 : i32
      %cst_npu_11 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_6, %cst_npu_7, %cst_npu_8, %cst_npu_9, %cst_npu_10, %cst_npu_11) : i32, i32, i32, i32, i32, i32

      // wait for dma output
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
      aiex.npu.dma_wait {symbol = @out2}
      aiex.npu.dma_wait {symbol = @out3}
    }
  }
}
