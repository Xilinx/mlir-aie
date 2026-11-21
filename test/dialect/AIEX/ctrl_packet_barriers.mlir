// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file --verify-diagnostics | FileCheck %s
// RUN: split-file %s %t
// RUN: aie-opt %t/normalize.mlir -aie-reorder-ctrl-packets-by-tile | FileCheck %s --check-prefix=NORMALIZE
// RUN: aie-opt %t/effects.mlir -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile -aie-coalesce-control-packets | FileCheck %s --check-prefix=EFFECTS
// RUN: aie-opt %t/multicast.mlir -aie-insert-ctrl-packet-barriers -aie-dedup-ctrl-packet-multicast | FileCheck %s --check-prefix=MULTICAST
// RUN: aie-opt %t/multicast.mlir -aie-insert-ctrl-packet-barriers -aie-dedup-ctrl-packet-multicast -aie-dedup-ctrl-packet-multicast | FileCheck %s --check-prefix=MULTICAST

// CHECK: aiex.control_packet_barrier {compiler_only, scope = {core = [#aie.tile<0, 2>], memory = [#aie.tile<1, 2>]}}
module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_barrier {compiler_only, scope = {core = [#aie.tile<0, 2>], memory = [#aie.tile<1, 2>]}}
      aiex.control_packet_barrier {compiler_only}
      aiex.control_packet_barrier
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{packet_effect requires compiler_only}}
      aiex.control_packet_barrier {packet_effect}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{scope requires compiler_only}}
      aiex.control_packet_barrier {scope = {memory = [#aie.tile<0, 2>]}}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{scope must contain physical modules}}
      aiex.control_packet_barrier {compiler_only, scope = {}}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{unknown scope module: other}}
      aiex.control_packet_barrier {compiler_only, scope = {other = [#aie.tile<0, 2>]}}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{scope modules require nonempty tile arrays}}
      aiex.control_packet_barrier {compiler_only, scope = {memory = []}}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{scope requires valid physical tiles}}
      aiex.control_packet_barrier {compiler_only, scope = {memory = [#aie.tile<99, 2>]}}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{core scope requires compute tiles}}
      aiex.control_packet_barrier {compiler_only, scope = {core = [#aie.tile<0, 1>]}}
    }
  }
}

// -----
//--- normalize.mlir

// NORMALIZE-LABEL: @independent
// NORMALIZE-NEXT: aiex.control_packet_barrier
// NORMALIZE-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>
// NORMALIZE-NEXT: aiex.control_packet {address = 2228228 : ui32, data = array<i32: 2>
// NORMALIZE-LABEL: @dependent
// NORMALIZE-NEXT: aiex.control_packet {address = 2097152 : ui32, data = array<i32: 1>
// NORMALIZE-NEXT: aiex.control_packet_barrier
// NORMALIZE-NEXT: aiex.control_packet {address = 2097152 : ui32, data = array<i32: 2>
module {
  aie.device(npu2) {
    aie.runtime_sequence @independent() {
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier {compiler_only, scope = {memory = [#aie.tile<0, 2>]}}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @dependent() {
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier {compiler_only, scope = {memory = [#aie.tile<0, 2>]}}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----
//--- effects.mlir

// EFFECTS-LABEL: @neighbor_acquire
// EFFECTS: data = array<i32: 3>
// EFFECTS: scope = {core = [#aie.tile<1, 2>], memory = [#aie.tile<0, 2>, #aie.tile<1, 2>, #aie.tile<1, 3>]}
// EFFECTS: address = 35856384 : ui32, data = array<i32: 1>
// EFFECTS: data = array<i32: 4>
// EFFECTS-LABEL: @independent_program
// EFFECTS: address = 2301952 : ui32, data = array<i32: 0>
// EFFECTS: address = 35782656 : ui32, data = array<i32: 2>
// EFFECTS-LABEL: @actions
// EFFECTS: address = 2219540 : ui32, data = array<i32: 1>
// EFFECTS: address = 2219540 : ui32, data = array<i32: 1>
// EFFECTS: address = 2301952 : ui32, data = array<i32: 2>
// EFFECTS: address = 2301952 : ui32, data = array<i32: 0>
// EFFECTS-LABEL: @unknown
// EFFECTS-NEXT: aiex.control_packet_barrier {compiler_only, packet_effect}
// EFFECTS-LABEL: @neighbor_release
// EFFECTS: address = 2097152 : ui32, data = array<i32: 7>
// EFFECTS: address = 3272704 : ui32, data = array<i32: 1>
// EFFECTS: address = 2097156 : ui32, data = array<i32: 9>
module {
  aie.device(npu2) {
    aie.runtime_sequence @neighbor_acquire() {
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @independent_program() {
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @actions() {
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @unknown() {
      aiex.control_packet {address = 3080192 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @neighbor_release() {
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3272704 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097156 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----
//--- multicast.mlir

module {
  aie.device(npu2) @overlay {
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
  } {overlay = @overlay}
  aie.device(npu2) @main {
    // MULTICAST-LABEL: @setup
    // MULTICAST: address = 2301952 : ui32, data = array<i32: 0>, multicast_id = 28
    // MULTICAST: address = 2219536 : ui32, data = array<i32: 2>, multicast_id = 28
    // MULTICAST: address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28
    // MULTICAST: address = 2219540 : ui32, data = array<i32: 1>, opcode
    // MULTICAST-NOT: aiex.control_packet {
    // MULTICAST: {{^ *}}}
    aie.runtime_sequence @setup() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219536 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3268112 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // MULTICAST-LABEL: @active
    // MULTICAST-NOT: multicast_id
    // MULTICAST: address = 2301952 : ui32, data = array<i32: 1>, opcode
    // MULTICAST-NOT: multicast_id
    // MULTICAST: address = 2097152 : ui32, data = array<i32: 7>, opcode
    // MULTICAST-NOT: multicast_id
    // MULTICAST: address = 3350528 : ui32, data = array<i32: 1>, opcode
    // MULTICAST-NOT: multicast_id
    // MULTICAST: address = 3145728 : ui32, data = array<i32: 7>, opcode
    aie.runtime_sequence @active() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3145728 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // MULTICAST-LABEL: @fence
    // MULTICAST: data = array<i32: 1>, multicast_id = 28
    // MULTICAST: aiex.control_packet_barrier {compiler_only, scope = {core = [#aie.tile<0, 2>, #aie.tile<0, 3>]}}
    // MULTICAST-NOT: multicast_id
    // MULTICAST: address = 2228228 : ui32, data = array<i32: 2>, opcode
    // MULTICAST-NOT: multicast_id
    // MULTICAST: address = 3276804 : ui32, data = array<i32: 2>, opcode
    aie.runtime_sequence @fence() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier {compiler_only, scope = {core = [#aie.tile<0, 2>, #aie.tile<0, 3>]}}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276804 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // MULTICAST-LABEL: @affinity_wait
    // MULTICAST: address = 2097152 : ui32, data = array<i32: 7>, multicast_id = 28
    // MULTICAST-NEXT: aiex.control_packet_barrier{{$}}
    // MULTICAST: address = 4399104 : ui32, data = array<i32: 1>, opcode
    aie.runtime_sequence @affinity_wait() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3145728 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 4399104 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}