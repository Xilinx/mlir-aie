//===- overlapping_rules_reject.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-verify-packet-rules %s

// The switchbox from Xilinx/mlir-aie#437. A switch matches rules in order and
// routes on the first hit, so an id matching both goes West by rule 1 and the
// South rule never sees it. Ids 10, 11, 14 and 15 are shared; the packet_flow
// ops make 10 and 14 live, and the diagnostic names the lowest live one.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    aie.packet_flow(10) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, West : 0> }
    aie.packet_flow(14) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, South : 0> }
    %sb = aie.switchbox(%tile) {
      %west = aie.amsel<0> (0)
      %south = aie.amsel<0> (1)
      %0 = aie.masterset(South : 0, %south)
      %1 = aie.masterset(West : 0, %west)
      aie.packet_rules(DMA : 0) {
        // expected-remark@+1 {{this is the rule that claims packet id 10}}
        aie.rule(26, 10, %west)
        // expected-error@+1 {{'aie.rule' op is shadowed for packet id 10}}
        aie.rule(24, 8, %south)
      }
    }
  }
}

// -----

// Relaxed masks that overlap only on ids nothing sends are how the router fits
// a port's flows into its slots -- id 18 is claimed by both rules and by no
// flow, which is what --aie-create-pathfinder-flows emits for live ids
// {10, 22} -> North:1 and {16, 19} -> North:4.
module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    aie.packet_flow(10) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(22) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(16) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 1> }
    aie.packet_flow(19) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 1> }
    %sb = aie.switchbox(%t00) {
      %up = aie.amsel<0> (0)
      %mem = aie.amsel<1> (0)
      %0 = aie.masterset(North : 1, %up)
      %1 = aie.masterset(North : 4, %mem)
      aie.packet_rules(South : 3) {
        aie.rule(3, 2, %up)
        aie.rule(28, 16, %mem)
      }
    }
  }
}

// -----

// Exact masks cannot overlap at all.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    aie.packet_flow(10) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, West : 0> }
    aie.packet_flow(14) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, South : 0> }
    %sb = aie.switchbox(%tile) {
      %west = aie.amsel<0> (0)
      %south = aie.amsel<0> (1)
      %0 = aie.masterset(South : 0, %south)
      %1 = aie.masterset(West : 0, %west)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 10, %west)
        aie.rule(31, 14, %south)
      }
    }
  }
}

// -----

// Rules naming the same amsel take the same route, so an overlap between them
// is redundant rather than a misroute. Here two distinct AMSelOps spell the
// same (arbiter, msel).
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    aie.packet_flow(10) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, West : 0> }
    %sb = aie.switchbox(%tile) {
      %a = aie.amsel<0> (0)
      %b = aie.amsel<0> (0)
      %0 = aie.masterset(West : 0, %a)
      %1 = aie.masterset(South : 0, %b)
      aie.packet_rules(DMA : 0) {
        aie.rule(30, 10, %a)
        aie.rule(31, 10, %b)
      }
    }
  }
}

// -----

// A rule whose value has bits outside its mask matches nothing: (id & 24) is
// one of {0, 8, 16, 24} and never 10, so it cannot shadow the rule after it.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    aie.packet_flow(8) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, South : 0> }
    %sb = aie.switchbox(%tile) {
      %west = aie.amsel<0> (0)
      %south = aie.amsel<0> (1)
      %0 = aie.masterset(South : 0, %south)
      %1 = aie.masterset(West : 0, %west)
      aie.packet_rules(DMA : 0) {
        aie.rule(24, 10, %west)
        aie.rule(24, 8, %south)
      }
    }
  }
}

// -----

// A hand-routed design names its ids in the DMA descriptors rather than in
// packet_flow ops: aie.dma_bd's packet attribute is a live-id source too.
module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %buf = aie.buffer(%t02) {sym_name = "buf"} : memref<8xi32>
    %l0 = aie.lock(%t02, 0) {init = 1 : i32}
    %l1 = aie.lock(%t02, 1) {init = 0 : i32}
    %sb = aie.switchbox(%t02) {
      %n = aie.amsel<0> (0)
      %s = aie.amsel<0> (1)
      %0 = aie.masterset(North : 0, %n)
      %1 = aie.masterset(South : 0, %s)
      aie.packet_rules(DMA : 0) {
        // expected-remark@+1 {{this is the rule that claims packet id 10}}
        aie.rule(26, 10, %n)
        // expected-error@+1 {{'aie.rule' op is shadowed for packet id 10}}
        aie.rule(24, 8, %s)
      }
    }
    %mem = aie.mem(%t02) {
      %d = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<8xi32>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
      aie.use_lock(%l1, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// An id named only by aiex.npu.dma_memcpy_nd in the runtime sequence, which is
// where a hand-routed shim design puts it.
module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %sb = aie.switchbox(%t02) {
      %n = aie.amsel<0> (0)
      %s = aie.amsel<0> (1)
      %0 = aie.masterset(North : 0, %n)
      %1 = aie.masterset(South : 0, %s)
      aie.packet_rules(DMA : 0) {
        // expected-remark@+1 {{this is the rule that claims packet id 14}}
        aie.rule(26, 10, %n)
        // expected-error@+1 {{'aie.rule' op is shadowed for packet id 14}}
        aie.rule(24, 8, %s)
      }
    }
    aie.shim_dma_allocation @in (%t00, MM2S, 0)
    aie.runtime_sequence @seq(%a: memref<8xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1], packet = <pkt_id = 14, pkt_type = 0>) {id = 0 : i64, metadata = @in} : memref<8xi32>
    }
  }
}

// -----

// An id named only by aiex.npu.writebd, the form a fully hand-written runtime
// sequence uses.
module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %sb = aie.switchbox(%t02) {
      %n = aie.amsel<0> (0)
      %s = aie.amsel<0> (1)
      %0 = aie.masterset(North : 0, %n)
      %1 = aie.masterset(South : 0, %s)
      aie.packet_rules(DMA : 0) {
        // expected-remark@+1 {{this is the rule that claims packet id 14}}
        aie.rule(26, 10, %n)
        // expected-error@+1 {{'aie.rule' op is shadowed for packet id 14}}
        aie.rule(24, 8, %s)
      }
    }
    aie.runtime_sequence @seq() {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 8 : i32, buffer_offset = 0 : i32,
        column = 0 : i32, row = 0 : i32, enable_packet = 1 : i32, out_of_order_id = 0 : i32,
        packet_id = 14 : i32, packet_type = 0 : i32,
        d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_before = 0 : i32, d0_zero_after = 0 : i32,
        d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_before = 0 : i32, d1_zero_after = 0 : i32,
        d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_before = 0 : i32, d2_zero_after = 0 : i32,
        iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32,
        lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32,
        lock_rel_id = 0 : i32, lock_rel_val = 0 : i32,
        next_bd = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----

// Overlapping rules that literally share one amsel value.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    aie.packet_flow(10) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, West : 0> }
    %sb = aie.switchbox(%tile) {
      %a = aie.amsel<0> (0)
      %0 = aie.masterset(West : 0, %a)
      aie.packet_rules(DMA : 0) {
        aie.rule(30, 10, %a)
        aie.rule(31, 10, %a)
      }
    }
  }
}

// -----

// Nothing in the device names a packet id, so there is nothing to check even
// though the rules overlap.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    %sb = aie.switchbox(%tile) {
      %west = aie.amsel<0> (0)
      %south = aie.amsel<0> (1)
      %0 = aie.masterset(South : 0, %south)
      %1 = aie.masterset(West : 0, %west)
      aie.packet_rules(DMA : 0) {
        aie.rule(26, 10, %west)
        aie.rule(24, 8, %south)
      }
    }
  }
}
