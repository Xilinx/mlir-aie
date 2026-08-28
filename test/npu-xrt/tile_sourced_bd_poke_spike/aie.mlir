//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Spike: can a compute tile's own core re-arm the SAME DMA BD twice, via a
// raw register write to its own MM2S channel's task-queue register, instead
// of either (a) one static aie.dma_bd per round (the working but
// BD-table-limited mechanism ProgramMemorySlot(source=...) uses today) or
// (b) BdIteration (hardware-verified to corrupt the packet's embedded
// address after the first repeat -- see
// ../tile_sourced_ctrl_pkt_spike/aie.mlir's lesson 3)?
//
// This is a minimal diff from tile_sourced_ctrl_pkt_spike/aie.mlir: same
// four "real" cores (0,2)/(0,3)/(0,4)/(0,5), same packet_flow/flow wiring,
// same expected output. Only tile_1_2's mem/core blocks change: instead of
// two chained aie.dma_bd entries (one per lock, as the working spike does),
// there is ONE aie.dma_bd (bd_id 0), executed twice -- once via the normal
// hardware-automatic initial queue push (round 0, lock0's write), and once
// via `push_bd0` (round 1, lock2's write): a compiled stub that writes BD 0's
// number directly to the MM2S channel 0 task-queue register
// (XAIE2PGBL_MEMORY_MODULE_DMA_MM2S_0_START_QUEUE, local offset 0x1DE14).
//
// RESOLVED (partially) 2026-08-24, hardware-verified (Strix): the core
// hypothesis holds -- a core-issued queue push IS a genuine descriptor
// fetch, re-framing the packet correctly each time, unlike BdIteration
// (which never re-fetches between repeats). tile_0_5 comes up with correct
// add-one output, proving lock2's write (round 1, via push_bd0()) lands
// correctly on top of lock0's (round 0).
//
// UPDATE 2026-08-25: pacing between rounds IS solved now, but not by this
// mechanism. This spike originally stopped here because two pacing attempts
// hung: waiting on the BD's own `Release(pkt8_prod_lock)` after round 0's
// send, and polling XAIE2PGBL_MEMORY_MODULE_DMA_MM2S_STATUS_0 for
// TaskQueueSize/ChannelRunning both zero. Both share the same root cause:
// the trailing lock-release genuinely does not fire reliably for a channel
// that terminates (`next_bd -> aie.end`) after a single non-looping BD --
// confirmed by isolating a next_bd change alone: switching this same BD's
// `next_bd ^bb2` to `next_bd ^bb1` (a plain self-loop back to its own
// block, the ordinary "keep streaming" idiom, and the documented default
// for `iron.overlay.Bd.next`) made the lock-release pacing work immediately
// -- no register poke of any kind needed. A self-loop is architecturally
// the same kind of thing as chaining to a *different* BD (an ordinary
// descriptor-chain traversal, a real fetch every hop) rather than
// `BdIteration` (which deliberately never re-fetches) -- so it reframes
// each round's packet correctly for the same underlying reason a poked
// re-arm does, without the poke, and with a completion signal that
// actually fires. `ProgramMemorySlot._load_tile_sourced` now uses exactly
// this (one BD, `next="self"`, core paces itself on an ordinary lock
// hand-off) and is hardware-verified at ~8 KB
// (test/npu-xrt/program_memory_overlay/hw/tile_sourced_iron_api.lit).
//
// The register-poke mechanism this file explores is hardware-verified to
// work as a *send* primitive (round 1 here does land correctly), but is not
// used in the shipped fix -- next="self" is simpler and its pacing is
// provably reliable, where the poked version's never was. Kept here as a
// record of what was tried and why it was set aside, not as something to
// build on.
//
// This spike has no run.lit: round 1's timing isn't guaranteed (see above),
// so wiring it into automatic test execution would be a flaky CI test, not
// a real regression gate. Reproduce by hand: compile push_bd0.cc for aie2p,
// sed NPUDEVICE -> npu2_4col in this file, run through %aiecc, then test.cpp
// against real Strix hardware (see ../tile_sourced_ctrl_pkt_spike/run.lit
// for the exact command shapes).
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

    // ONE BD's worth of payload (a single lock write: header + data), reused
    // for both rounds -- not one buffer per round.
    %pkt8_payload = aie.buffer(%tile_1_2) {sym_name = "pkt8_payload"} : memref<2xi32>
    // prod_lock: "the core may (over)write pkt8_payload and arm a send."
    // Starts available (init=1) so round 0 can proceed immediately; the BD
    // gives it back (Release) once a send has actually drained the buffer,
    // gating round 1 on round 0 having actually completed.
    %pkt8_prod_lock = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "pkt8_prod_lock"}
    // cons_lock: "pkt8_payload holds a real, unsent packet; go." The BD
    // acquires this before every send, whether the send is the automatic
    // initial queue push (round 0) or a manual push_bd0() (round 1).
    %pkt8_cons_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "pkt8_cons_lock"}

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
    aie.packet_flow(0x8) {
      aie.packet_source<%tile_1_2, DMA : 0>
      aie.packet_dest<%tile_0_5, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 1)
    aie.flow(%tile_0_3, DMA : 0, %tile_1_0, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %tile_1_0, DMA : 1)
    aie.flow(%tile_0_5, DMA : 0, %tile_2_0, DMA : 0)

    func.func private @push_bd0() attributes {link_with = "push_bd0.o"}

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

    // tile_1_2's core: two rounds, one BD, reused. Round 0 (lock0's write)
    // rides the automatic initial queue push dma_start already gives bd_id 0
    // -- no new mechanism, identical to the working spike's first send.
    // Round 1 (lock2's write) is the actual test: the channel already went
    // idle after round 0 (next_bd -> ^end, no hardware auto-chain), so
    // nothing re-executes bd_id 0 unless something explicitly re-arms it --
    // that "something" is push_bd0(), a raw register write, not next_bd and
    // not BdIteration.
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %h0 = arith.constant 0x0001F000 : i32
      %h1 = arith.constant 0x8001F020 : i32
      %d = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32

      // round 0: lock0's write.
      aie.use_lock(%pkt8_prod_lock, AcquireGreaterEqual, %c1_i32)
      memref.store %h0, %pkt8_payload[%c0] : memref<2xi32>
      memref.store %d, %pkt8_payload[%c1] : memref<2xi32>
      aie.use_lock(%pkt8_cons_lock, Release, %c1_i32)

      // DEBUG ISOLATION F: no wait before round 1 -- race, but tests
      // whether push_bd0() can ever deliver a second packet at all.
      memref.store %h1, %pkt8_payload[%c0] : memref<2xi32>
      memref.store %d, %pkt8_payload[%c1] : memref<2xi32>
      aie.use_lock(%pkt8_cons_lock, Release, %c1_i32)
      func.call @push_bd0() : () -> ()

      aie.end
    }

    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%pkt8_cons_lock, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%pkt8_payload : memref<2xi32> offset = 0 len = 2) { bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 1, pkt_id = 8> }
      aie.use_lock(%pkt8_prod_lock, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @ctrlin0 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @out1 (%tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @out2 (%tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @out3 (%tile_2_0, S2MM, 0)

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
