//===- auto-packetize-union-k.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two heterogeneous npu2 configs share column 0. Each config saturates both
// shim MM2S channels with circuit ingress, so the auto-packetize pass must pick
// ONE control trunk channel K per column that is consistent across all configs
// and stamp it on every config's column-0 shim tile.

// RUN: aie-opt -aie-auto-packetize-control-ingress -split-input-file -verify-diagnostics %s | FileCheck %s
// RUN: not aie-opt -aie-auto-packetize-control-ingress -split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK: aie.tile(0, 0) {{.*}}ctrl_pkt_trunk_chan = [[K:[0-9]+]]
// CHECK: aie.tile(0, 0) {{.*}}ctrl_pkt_trunk_chan = [[K]]
module {
  aie.device(npu2) @cfg_a {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @a0(%sh, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-warning @below {{auto-packetized objectFifo 'a1' on column 0 from circuit -> packet for resident control coexistence}}
    aie.objectfifo @a1(%sh, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
  aie.device(npu2) @cfg_b {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @b0(%sh, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-warning @below {{auto-packetized objectFifo 'b1' on column 0 from circuit -> packet for resident control coexistence}}
    aie.objectfifo @b1(%sh, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Union semantics: config A's manual aie.shim_mux blocks channel 0, config B is
// free on channel 0. A per-device choice would leave B on channel 0; the UNION
// must shift both configs to channel 1 (the only channel free in EVERY config).
// Assert BOTH column-0 shim tiles land on the SAME K, and that K is 1 -- this
// case fails if chooseUnionTrunkChan regresses to per-device selection.

// CHECK: aie.tile(0, 0) {{.*}}ctrl_pkt_trunk_chan = 1 : i32
// CHECK: aie.tile(0, 0) {{.*}}ctrl_pkt_trunk_chan = 1 : i32
module {
  aie.device(npu2) @union_a {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    aie.shim_mux(%sh) {
      aie.connect<DMA : 0, North : 3>
    }
    aie.objectfifo @ua0(%sh, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
  aie.device(npu2) @union_b {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    aie.objectfifo @ub0(%sh, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Two configs whose manual aie.shim_mux occupy OPPOSITE shim MM2S channels on
// column 0: config A blocks channel 0, config B blocks channel 1. Their union
// leaves no channel free of manual routing, so no consistent control trunk
// exists for column 0 and the pass must fail loud.

// ERR: no shim mm2s channel is free of manual routing across all configs for column 0
module {
  // expected-error @below {{no shim mm2s channel is free of manual routing across all configs for column 0}}
  aie.device(npu2) @cfg_c {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    aie.shim_mux(%sh) {
      aie.connect<DMA : 0, North : 3>
    }
    aie.objectfifo @c0(%sh, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
  aie.device(npu2) @cfg_d {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    aie.shim_mux(%sh) {
      aie.connect<DMA : 1, North : 3>
    }
    aie.objectfifo @d0(%sh, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
