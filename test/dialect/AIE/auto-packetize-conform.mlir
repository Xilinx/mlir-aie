//===- auto-packetize-conform.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-auto-packetize-control-ingress -split-input-file %s | FileCheck %s

// Two heterogeneous npu2 configs share column 0. Both saturate the two shim MM2S
// channels with circuit ingress, but the least-disruptive leg differs per config:
// config A's smaller leg is @a0, config B's smaller leg is @b1. Absent
// coordination the two would packetize opposite-position legs and scatter their
// control trunks to different channels. The conform decision forces BOTH configs
// onto the same union trunk K: the packetized leg of EACH config is pinned to K,
// and the remaining circuit leg is pinned to the other channel. The [[K]] backref
// ties both packet legs to the identical channel; the circuit legs land on 1.
// Each packetized single-hop leg is confined to its shim hop via a synthesized
// memtile relay (@*_relay + aie.objectfifo.link).
// CHECK: aie.objectfifo @a0{{.*}}{packet, prod_dma_channel = [[K:[0-9]+]] : i32}
// CHECK: aie.objectfifo @a0_relay
// CHECK: aie.objectfifo.link [@a0] -> [@a0_relay]
// CHECK: aie.objectfifo @a1{{.*}}prod_dma_channel = 1 : i32
// CHECK: aie.objectfifo @b0{{.*}}prod_dma_channel = 1 : i32
// CHECK: aie.objectfifo @b1{{.*}}{packet, prod_dma_channel = [[K]] : i32}
// CHECK: aie.objectfifo @b1_relay
// CHECK: aie.objectfifo.link [@b1] -> [@b1_relay]
module {
  aie.device(npu2) @cfg_a {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @a0(%sh, {%a}, 2 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @a1(%sh, {%b}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
  }
  aie.device(npu2) @cfg_b {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @b0(%sh, {%a}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b1(%sh, {%b}, 2 : i32) : !aie.objectfifo<memref<4xi32>>
  }
}

// -----

// Swap before packetize: config already has a packet leg @s0 and a circuit leg
// @s1 on column 0. The conform decision swaps the EXISTING packet leg onto K and
// pins the circuit leg off K -- it demands no NEW channel. @s0 is single-hop, so
// its packet segment is still confined to the shim hop via a memtile relay
// (@s0_relay + link). @s1 must stay circuit: its attr dict opens directly with
// prod_dma_channel (no {packet, ...}), so this line fails if @s1 were packetized.
// CHECK: aie.objectfifo @s0{{.*}}{packet, prod_dma_channel = 0 : i32}
// CHECK: aie.objectfifo @s0_relay
// CHECK: aie.objectfifo.link [@s0] -> [@s0_relay]
// CHECK: aie.objectfifo @s1{{.*}}) {prod_dma_channel = 1 : i32}
module {
  aie.device(npu2) @swap {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @s0(%sh, {%a}, 2 : i32) {packet} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @s1(%sh, {%b}, 2 : i32) : !aie.objectfifo<memref<4xi32>>
  }
}

// -----

// Least-disruptive by FAN-OUT: two circuit legs on column 0 where the
// SMALLER-data leg @m0 has HIGHER fan-out (2 consumer tiles) and the LARGER-data
// leg @m1 has LOWER fan-out (1 consumer tile). The conform decision packetizes
// the FEWEST-fan-out leg @m1 onto K, because fan-out -- not data volume -- drives
// shim slave-port packet-rule slot pressure (the exact thing that overflowed on
// ml/bottleneck). A data-volume metric would wrongly pick the smaller @m0; this
// case fails under that metric. @m0 stays circuit: its attr dict opens directly
// with prod_dma_channel, so that line fails if @m0 were wrongly packetized.
// CHECK: aie.objectfifo @m0{{.*}}) {prod_dma_channel = 1 : i32}
// CHECK: aie.objectfifo @m1{{.*}}{packet, prod_dma_channel = 0 : i32}
// CHECK: aie.objectfifo @m1_relay
// CHECK: aie.objectfifo.link [@m1] -> [@m1_relay]
module {
  aie.device(npu2) @least_disruptive {
    %sh = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @m0(%sh, {%a, %b}, 2 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @m1(%sh, {%c}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
  }
}
