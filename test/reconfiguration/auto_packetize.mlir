//===- auto_packetize.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-auto-packetize-shim-input --split-input-file --verify-diagnostics %s | FileCheck %s

// Two circuit inputs hold both shim MM2S channels of column 0. The one with
// the fewest consumers (a tie, so the last name) becomes packet-switched on
// channel 0, relayed by the memory tile; the other moves to channel 1.

// CHECK-LABEL: @saturated
// CHECK:       aie.tile(0, 0) {ctrl_pkt_shim_chan = 0 : i32}
// CHECK:       aie.objectfifo @in0(%{{.+}}, {%[[CORE:.+]]}, 1 : i32) {prod_dma_channel = 1 : i32}
// CHECK:       aie.objectfifo @in1(%{{.+}}, {%[[MEM:.+]]}, 1 : i32) {packet, prod_dma_channel = 0 : i32}
// CHECK:       aie.objectfifo @in1_relay(%[[MEM]], {%[[CORE]]}, 1 : i32)
// CHECK:       aie.objectfifo.link [@in1] -> [@in1_relay]([] [0])
// CHECK:       aie.objectfifo.acquire @in1_relay(Consume, 1)
// CHECK:       aie.objectfifo.release @in1_relay(Consume, 1)
module {
  aie.device(npu2) @saturated {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in0 (%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    // expected-remark@+1 {{auto-packetized for control on shim MM2S channel 0 of column 0}}
    aie.objectfifo @in1 (%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.core(%core) {
      %a = aie.objectfifo.acquire @in0 (Consume, 1) : memref<4xi32>
      %b = aie.objectfifo.acquire @in1 (Consume, 1) : memref<4xi32>
      aie.objectfifo.release @in0 (Consume, 1)
      aie.objectfifo.release @in1 (Consume, 1)
      aie.end
    }
  }
}

// -----

// The input with fewer consumers is the one packetized. An input that ends at
// the memory tile needs no relay.

// CHECK-LABEL: @fewest_consumers
// CHECK:       aie.objectfifo @wide(%{{.+}}, {%{{.+}}, %{{.+}}}, 1 : i32) {prod_dma_channel = 1 : i32}
// CHECK:       aie.objectfifo @narrow(%{{.+}}, {%[[MEM:.+]]}, 1 : i32) {packet, prod_dma_channel = 0 : i32}
// CHECK-NOT:   _relay
module {
  aie.device(npu2) @fewest_consumers {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %core3 = aie.tile(0, 3)
    aie.objectfifo @wide (%shim, {%core, %core3}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    // expected-remark@+1 {{auto-packetized for control on shim MM2S channel 0 of column 0}}
    aie.objectfifo @narrow (%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
  }
}

// -----

// A column with a channel left free is not touched.

// CHECK-LABEL: @free_channel
// CHECK-NOT:   ctrl_pkt_shim_chan
// CHECK-NOT:   prod_dma_channel
// CHECK-NOT:   packet
module {
  aie.device(npu2) @free_channel {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in0 (%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
  }
}

// -----

// An input that is already packet-switched takes the control channel as it
// is; no input is packetized or relayed.

// CHECK-LABEL: @packet_input
// CHECK:       aie.objectfifo @a(%{{.+}}, {%{{.+}}}, 1 : i32) {prod_dma_channel = 1 : i32}
// CHECK:       aie.objectfifo @b(%{{.+}}, {%{{.+}}}, 1 : i32) {packet, prod_dma_channel = 0 : i32}
// CHECK-NOT:   aie.objectfifo.link
module {
  aie.device(npu2) @packet_input {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @a (%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @b (%shim, {%core}, 1 : i32) {packet} : !aie.objectfifo<memref<4xi32>>
  }
}

// -----

// No device fills column 0 alone, but @one takes channel 0 in @first and
// @second's flow holds channel 1 as a circuit, so together they hold both
// channels. Control goes on channel 0, and @one moves to channel 1.

// CHECK-LABEL: @first
// CHECK:       aie.tile(0, 0) {ctrl_pkt_shim_chan = 0 : i32}
// CHECK:       aie.objectfifo @one(%{{.+}}, {%{{.+}}}, 1 : i32) {prod_dma_channel = 1 : i32}
// CHECK-LABEL: @second
// CHECK:       aie.tile(0, 0) {ctrl_pkt_shim_chan = 0 : i32}
module {
  aie.device(npu2) @first {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @one (%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
  }
  aie.device(npu2) @second {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.flow(%shim, DMA : 1, %mem, DMA : 0)
  }
}

// -----

// Each saturated column gets its own control channel.

// CHECK-LABEL: @two_columns
// CHECK-DAG:   aie.tile(0, 0) {ctrl_pkt_shim_chan = 0 : i32}
// CHECK-DAG:   aie.tile(1, 0) {ctrl_pkt_shim_chan = 0 : i32}
// CHECK-DAG:   aie.objectfifo @a1(%{{.+}}, {%{{.+}}}, 1 : i32) {packet, prod_dma_channel = 0 : i32}
// CHECK-DAG:   aie.objectfifo @b1(%{{.+}}, {%{{.+}}}, 1 : i32) {packet, prod_dma_channel = 0 : i32}
module {
  aie.device(npu2) @two_columns {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %mem0 = aie.tile(0, 1)
    %mem1 = aie.tile(1, 1)
    aie.objectfifo @a0 (%shim0, {%mem0}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    // expected-remark@+1 {{auto-packetized for control on shim MM2S channel 0 of column 0}}
    aie.objectfifo @a1 (%shim0, {%mem0}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @b0 (%shim1, {%mem1}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    // expected-remark@+1 {{auto-packetized for control on shim MM2S channel 0 of column 1}}
    aie.objectfifo @b1 (%shim1, {%mem1}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
  }
}

// -----

// Circuit flows hold both channels, so control has none to share.

module {
  // expected-error@+1 {{column 0 has no shim MM2S channel control packets can share: every channel carries circuit-switched data}}
  aie.device(npu2) @no_shareable_channel {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%shim, DMA : 1, %mem, DMA : 1)
    aie.objectfifo @in0 (%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
  }
}

// -----

// Three inputs for two channels. One shares the control channel, and the other
// two can't both have the remaining one.

module {
  aie.device(npu2) @too_many_inputs {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.objectfifo @a (%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    // expected-error@+1 {{column 0 has no shim MM2S channel left for this objectFIFO beside the control packets on channel 0}}
    aie.objectfifo @b (%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    // expected-remark@+1 {{auto-packetized for control on shim MM2S channel 0 of column 0}}
    aie.objectfifo @c (%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
  }
}
