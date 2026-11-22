// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file --verify-diagnostics | FileCheck %s
// RUN: split-file %s %t
// RUN: aie-opt %t/normalize.mlir -aie-reorder-ctrl-packets-by-tile | FileCheck %s --check-prefix=NORMALIZE
// RUN: aie-opt %t/effects.mlir -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile -aie-coalesce-control-packets | FileCheck %s --check-prefix=EFFECTS
// RUN: aie-opt %t/runtime_parameters.mlir -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile -aie-coalesce-control-packets | FileCheck %s --check-prefix=PARAMETER
// RUN: %aiecc -j1 --load-pdi-to-ctrl-pkt --ctrlpkt-optimize=false --get=npu_expanded.mlir --tmpdir=%t/no_opt --output-dir=%t/no_opt %t/runtime_parameters.mlir
// RUN: FileCheck %s --check-prefix=NOOPT < %t/no_opt/npu_expanded.mlir
// RUN: aie-opt %t/multicast.mlir -aie-insert-ctrl-packet-barriers -aie-dedup-ctrl-packet-multicast | FileCheck %s --check-prefix=MULTICAST
// RUN: aie-opt %t/multicast.mlir -aie-insert-ctrl-packet-barriers -aie-dedup-ctrl-packet-multicast -aie-dedup-ctrl-packet-multicast | FileCheck %s --check-prefix=MULTICAST
// RUN: aie-opt %t/substrings.mlir -aie-insert-ctrl-packet-barriers -aie-dedup-ctrl-packet-multicast | FileCheck %s --check-prefix=PREFIX
// RUN: aie-opt %t/substrings.mlir -aie-insert-ctrl-packet-barriers -aie-dedup-ctrl-packet-multicast=substrings=true -aie-check-ctrl-packet-delivery | FileCheck %s --check-prefix=SUBSTRING
// RUN: aie-opt %t/substrings.mlir -aie-insert-ctrl-packet-barriers -aie-dedup-ctrl-packet-multicast="substrings=true parallel-shims=true" -aie-check-ctrl-packet-delivery=parallel-shims=true | FileCheck %s --check-prefix=PARALLEL
// RUN: aie-opt %t/nested.mlir -aie-unroll-runtime-sequence-loops -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile -aie-dedup-ctrl-packet-multicast -aie-check-ctrl-packet-delivery | FileCheck %s --check-prefix=NESTED

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
  aie.device(npu2) @cfg {
    aie.runtime_sequence() {
      // expected-error @+1 {{drains requires a hardware barrier}}
      aiex.control_packet_barrier {compiler_only, drains = @cfg}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error @+1 {{@missing is not a device}}
      aiex.control_packet_barrier {drains = @missing}
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
// NORMALIZE-NEXT: aiex.set_device @normalization {quiescent}
// NORMALIZE-NEXT: aiex.control_packet_barrier
// NORMALIZE-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>
// NORMALIZE-NEXT: aiex.control_packet {address = 2228228 : ui32, data = array<i32: 2>
// NORMALIZE-LABEL: @dependent
// NORMALIZE-NEXT: aiex.set_device @normalization {quiescent}
// NORMALIZE-NEXT: aiex.control_packet {address = 2097152 : ui32, data = array<i32: 1>
// NORMALIZE-NEXT: aiex.control_packet_barrier
// NORMALIZE-NEXT: aiex.control_packet {address = 2097152 : ui32, data = array<i32: 2>
module {
  aie.device(npu2) @normalization {
    aie.runtime_sequence @independent() {
      aiex.set_device @normalization {quiescent}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier {compiler_only, scope = {memory = [#aie.tile<0, 2>]}}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @dependent() {
      aiex.set_device @normalization {quiescent}
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
// EFFECTS: address = 35856384 : ui32, data = array<i32: 1>
// EFFECTS: scope = {core = [#aie.tile<1, 2>], memory = [#aie.tile<0, 2>, #aie.tile<1, 2>, #aie.tile<1, 3>], switch =
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
  aie.device(npu2) @effects {
    aie.runtime_sequence @neighbor_acquire() {
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @independent_program() {
      aiex.set_device @effects {quiescent}
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
//--- runtime_parameters.mlir

module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2) @main {
    // PARAMETER-LABEL: @after_sync
    // PARAMETER: aiex.npu.sync
    // PARAMETER: address = 2097156 : ui32, data = array<i32: 7>
    // PARAMETER: address = 2097152 : ui32, data = array<i32: 1>
    // PARAMETER: address = 2097152 : ui32, data = array<i32: 2>
    // PARAMETER: {{^ *}}}
    aie.runtime_sequence @after_sync() {
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aiex.npu.sync(%zero, %zero, %zero, %zero, %one, %one) : i32, i32, i32, i32, i32, i32
      aiex.control_packet {address = 2097156 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // PARAMETER-LABEL: @inactive
    // PARAMETER: aiex.set_device @cfg {quiescent}
    // PARAMETER-NEXT: address = 2097152 : ui32, data = array<i32: 2, 7>
    // PARAMETER-NEXT: {{^ *}}}
    // NOOPT-LABEL: @inactive
    // NOOPT: address = 2097156 : ui32, data = array<i32: 7>
    // NOOPT: address = 2097152 : ui32, data = array<i32: 1>
    // NOOPT: address = 2097152 : ui32, data = array<i32: 2>
    // NOOPT-NOT: multicast_id
    // NOOPT: {{^ *}}}
    aie.runtime_sequence @inactive() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2097156 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // PARAMETER-LABEL: @switch_before_start
    // PARAMETER: address = 2355204 : ui32, data = array<i32: 7>
    // PARAMETER: address = 2219540 : ui32, data = array<i32: 1>
    // PARAMETER: address = 2228224 : ui32, data = array<i32: 9>
    aie.runtime_sequence @switch_before_start() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet_barrier
      aiex.control_packet {address = 2355204 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----
//--- nested.mlir

// NESTED-LABEL: aie.runtime_sequence @nested_parameters
// NESTED: aiex.set_device @cfg
// NESTED-NOT: scf.for
// NESTED: aiex.npu.sync
// NESTED-NOT: multicast_id
// NESTED: address = 2097152 : ui32, data = array<i32: 7>
// NESTED: address = 3145728 : ui32, data = array<i32: 7>
// NESTED: aiex.npu.sync
// NESTED: address = 2097152 : ui32, data = array<i32: 7>
// NESTED: address = 3145728 : ui32, data = array<i32: 7>
// NESTED-NOT: scf.for
// NESTED: {{^ *}}}
module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @nested_parameters() {
      %begin = arith.constant 0 : index
      %end = arith.constant 2 : index
      %step = arith.constant 1 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aiex.set_device @cfg
      scf.for %iteration = %begin to %end step %step {
        aiex.npu.sync(%zero, %zero, %zero, %zero, %one, %one) : i32, i32, i32, i32, i32, i32
        aiex.control_packet {address = 2097152 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3145728 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      }
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
    // MULTICAST-LABEL: @independent_program_after_start
    // MULTICAST: address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28
    // MULTICAST: address = 2219540 : ui32, data = array<i32: 1>, opcode
    // MULTICAST: address = 2097152 : ui32, data = array<i32: 8>, opcode
    // MULTICAST: address = 2097152 : ui32, data = array<i32: 9>, opcode
    aie.runtime_sequence @independent_program_after_start() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2097152 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
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

// -----
//--- substrings.mlir

// PREFIX-NOT: multicast_id
module {
  aie.device(npu2) @span {
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<1, 2>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @span_cfg {
  } {overlay = @span}
  aie.device(npu2) @overlay {
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
  } {overlay = @overlay}
  aie.device(npu2) @main {
    // SUBSTRING-LABEL: @internal
    // SUBSTRING: data = array<i32: 10>, opcode
    // SUBSTRING: aiex.control_packet_barrier{{$}}
    // SUBSTRING-NEXT: address = 2228228 : ui32, data = array<i32: 7, 8>, multicast_id = 28
    // SUBSTRING-NEXT: aiex.control_packet_barrier{{$}}
    // SUBSTRING: data = array<i32: 20>, opcode
    // SUBSTRING-NOT: multicast_id
    // SUBSTRING: {{^ *}}}
    aie.runtime_sequence @internal() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 10>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 20>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276804 : ui32, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // SUBSTRING-LABEL: @weighted
    // SUBSTRING: address = 2228228 : ui32, data = array<i32: 1>, opcode
    // SUBSTRING: address = 2228240 : ui32, data = array<i32: 7, 8, 9, 10>, multicast_id = 28
    // SUBSTRING-NOT: multicast_id
    // SUBSTRING: {{^ *}}}
    aie.runtime_sequence @weighted() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 10>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228232 : ui32, data = array<i32: 10>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228240 : ui32, data = array<i32: 7, 8, 9, 10>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 20>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276804 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276808 : ui32, data = array<i32: 20>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276816 : ui32, data = array<i32: 7, 8, 9, 10>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // SUBSTRING-LABEL: @ordered
    // SUBSTRING: address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28
    // SUBSTRING: address = 2219540 : ui32, data = array<i32: 1>, multicast_id = 28
    // SUBSTRING: address = 2219540 : ui32, data = array<i32: 1>, multicast_id = 28
    // SUBSTRING-NOT: multicast_id
    // SUBSTRING: {{^ *}}}
    aie.runtime_sequence @ordered() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2228240 : ui32, data = array<i32: 10>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276816 : ui32, data = array<i32: 20>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3268116 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3268116 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // SUBSTRING-LABEL: @fenced
    // SUBSTRING-NOT: multicast_id
    // SUBSTRING: {{^ *}}}
    aie.runtime_sequence @fenced() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 10>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier {compiler_only, scope = {core = [#aie.tile<0, 2>, #aie.tile<0, 3>]}}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 20>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276804 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // SUBSTRING-LABEL: @serial_columns
    // SUBSTRING-NOT: multicast_id
    // SUBSTRING: {{^ *}}}
    // PARALLEL-LABEL: @serial_columns
    // PARALLEL: address = 2228228 : ui32, data = array<i32: 7>, multicast_id = 28
    aie.runtime_sequence @serial_columns() {
      aiex.set_device @span_cfg {quiescent}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 10>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 20>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782660 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // SUBSTRING-LABEL: @sink
    // SUBSTRING: address = 2301952 : ui32, data = array<i32: 0>, opcode
    // SUBSTRING: address = 3350528 : ui32, data = array<i32: 2>, opcode
    // SUBSTRING: aiex.control_packet_barrier{{$}}
    // SUBSTRING-NEXT: address = 2228224 : ui32, data = array<i32: 7, 8>, multicast_id = 28
    // SUBSTRING-NEXT: aiex.control_packet_barrier{{$}}
    // SUBSTRING-NOT: multicast_id
    // SUBSTRING: {{^ *}}}
    aie.runtime_sequence @sink() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // SUBSTRING-LABEL: @overwritten_tail
    // SUBSTRING-NOT: multicast_id
    // SUBSTRING: {{^ *}}}
    aie.runtime_sequence @overwritten_tail() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}