//===- teardown.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="teardown=true" --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-expand-load-pdi="teardown=true init-entry=true" --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=INIT
// RUN: not aie-opt --aie-expand-load-pdi="init-entry=true" --split-input-file %s 2>&1 | FileCheck %s --check-prefix=NO-TEARDOWN

// NO-TEARDOWN: error: aie-expand-load-pdi: init-entry requires teardown

// @cfg sets a circuit and a packet route in tile (0, 2) and starts the
// memtile's MM2S channel 1. Its teardown, after the configuration's last sync,
// zeroes the circuit's master and slave ports (2355204, 2355480), the packet
// route's (2355200, 2355484) and its slot (2355824), and pulses the reset bit
// of the memtile channel (1705528). Only the first configuration keeps its reset, and nothing is
// appended.

// CHECK-LABEL: aie.runtime_sequence(%arg0: memref<16xi32>)
// CHECK:     %[[SYNC_ADDR:.*]] = arith.constant 119316 : i32
// CHECK:     aiex.npu.load_pdi {device_ref = @empty_0
// CHECK:     aiex.npu.write32(%[[SYNC_ADDR]], %{{.*}})
// CHECK:     aiex.npu.sync
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[A0:.*]] = arith.constant 2355204 : i32
// CHECK:     aiex.npu.write32(%[[A0]], %[[ZERO]])
// CHECK-DAG: %[[A1:.*]] = arith.constant 2355480 : i32
// CHECK:     aiex.npu.write32(%[[A1]], %[[ZERO]])
// CHECK-DAG: %[[A2:.*]] = arith.constant 2355200 : i32
// CHECK:     aiex.npu.write32(%[[A2]], %[[ZERO]])
// CHECK-DAG: %[[A3:.*]] = arith.constant 2355484 : i32
// CHECK:     aiex.npu.write32(%[[A3]], %[[ZERO]])
// CHECK-DAG: %[[A4:.*]] = arith.constant 2355824 : i32
// CHECK:     aiex.npu.write32(%[[A4]], %[[ZERO]])
// CHECK-DAG: %[[RESET:.*]] = arith.constant 1705528 : i32
// CHECK-DAG: %[[BIT:.*]] = arith.constant 2 : i32
// CHECK:     aiex.npu.maskwrite32(%[[RESET]], %[[BIT]], %[[BIT]])
// CHECK:     aiex.npu.maskwrite32(%[[RESET]], %[[ZERO]], %[[BIT]])
// CHECK:     aiex.npu.write32(%[[SYNC_ADDR]], %{{.*}})
// CHECK-NOT: aiex.npu.load_pdi
// CHECK:     aiex.npu.sync
// CHECK:     aiex.npu.write32(%{{.*}}, %{{.*}})
// CHECK:     aiex.npu.maskwrite32
// CHECK:     aiex.npu.maskwrite32
// CHECK-NOT: aiex.npu.load_pdi
// CHECK:     {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %mem = aie.tile(0, 1)
    %compute = aie.tile(0, 2)
    aie.switchbox(%compute) {
      aie.connect<South : 1, DMA : 0>
      %a = aie.amsel<0> (0)
      aie.masterset(Core : 0, %a)
      aie.packet_rules(South : 2) {
        aie.rule(31, 3, %a)
      }
    }
    %buf = aie.buffer(%mem) {address = 0 : i32, sym_name = "buf"} : memref<16xi32>
    %memdma = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %a = arith.constant 119316 : i32
      aiex.npu.load_pdi {device_ref = @cfg}
      aiex.npu.write32(%a, %c0) : i32, i32
      aiex.npu.sync(%c0, %c0, %c0, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.npu.write32(%a, %c1) : i32, i32
      aiex.npu.load_pdi {device_ref = @cfg}
      aiex.npu.sync(%c0, %c0, %c0, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// A configuration with no sync is torn down where the next one starts.

// CHECK-LABEL: aie.runtime_sequence(%arg0: memref<16xi32>)
// CHECK:     %[[ADDR:.*]] = arith.constant 119316 : i32
// CHECK:     aiex.npu.load_pdi {device_ref = @empty_0
// CHECK:     aiex.npu.write32(%[[ADDR]], %{{.*}})
// CHECK:     aiex.npu.maskwrite32(%[[RESET:.*]], %{{.*}}, %{{.*}})
// CHECK:     aiex.npu.maskwrite32(%[[RESET]], %{{.*}}, %{{.*}})
// CHECK-NOT: aiex.npu.load_pdi
// CHECK:     aiex.npu.blockwrite
// CHECK:     aiex.npu.maskwrite32(%[[RESET2:.*]], %{{.*}}, %{{.*}})
// CHECK:     aiex.npu.maskwrite32(%[[RESET2]], %{{.*}}, %{{.*}})
// CHECK-NOT: aiex.npu
// CHECK:     {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) {address = 0 : i32, sym_name = "buf"} : memref<16xi32>
    %memdma = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %a = arith.constant 119316 : i32
      aiex.npu.load_pdi {device_ref = @cfg}
      aiex.npu.write32(%a, %c0) : i32, i32
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// With an init entry, the reset moves into one `reconfig_init` sequence shared
// by both sequences.

// INIT-LABEL: aie.device(npu2_1col) @shared
// INIT:       aie.runtime_sequence @reconfig_init() {
// INIT-NEXT:    aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
// INIT-NEXT:  }
// INIT:       aie.runtime_sequence @first(%arg0: memref<16xi32>)
// INIT-NOT:   aiex.npu.load_pdi
// INIT:       aie.runtime_sequence @second(%arg0: memref<16xi32>)
// INIT-NOT:   aiex.npu.load_pdi
// INIT:       // -----

module {
  aie.device(npu2_1col) @shared {
    %compute = aie.tile(0, 2)
    aie.switchbox(%compute) {
      aie.connect<South : 1, DMA : 0>
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence @first(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @shared}
    }
    aie.runtime_sequence @second(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @shared}
    }
  }
}

// -----

// The init entry is named `reconfig_init`, so no other symbol of the device may
// be.

module {
  aie.device(npu2_1col) @cfg {
    %t = aie.tile(0, 0)
  }
  aie.device(npu2_1col) @main {
    // expected-error@+1 {{is named 'reconfig_init', which aie-expand-load-pdi needs for the device's init entry}}
    aie.runtime_sequence @reconfig_init(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// With control packet reconfiguration (expand_mode = 2), only the first
// configuration loads the overlay, or none with an init entry. The teardown
// keeps what the overlay sets, which is the memtile's control circuit, the
// TileControl master of tile (0, 2), and the overlay rule of slave port
// South : 3, where only the slot of the configuration's own rule (2355844) is
// disabled. The memtile channel's reset pulse keeps the enable bit the
// configuration set (1705528 = 1).

// CHECK-LABEL: aie.runtime_sequence(%arg0: memref<16xi32>)
// CHECK:       aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay,
// CHECK:       aiex.control_packet {address = 2355844 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 52363521>
// CHECK:       aiex.control_packet {address = 1705528 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>
// CHECK-NEXT:  aiex.npu.sync
// CHECK-NEXT:  aiex.control_packet {address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 0>
// CHECK-NEXT:  aiex.control_packet {address = 2355480 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 0>
// CHECK-NEXT:  aiex.control_packet {address = 2355200 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 0>
// CHECK-NEXT:  aiex.control_packet {address = 2355844 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 0>
// CHECK-NEXT:  aiex.control_packet {address = 1705528 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 3>
// CHECK-NEXT:  aiex.control_packet {address = 1705528 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>
// CHECK-NEXT:  aiex.control_packet {address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: -2147483642>
// CHECK-NOT:   aiex.npu.load_pdi
// CHECK:       aiex.npu.sync
// CHECK-NEXT:  aiex.control_packet {address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 0>
// CHECK-NOT:   aiex.npu.load_pdi
// CHECK:       {{^    }}}

// INIT-LABEL: aie.device(npu2_1col) @ctrl_pkt_overlay
// INIT:       aie.runtime_sequence @reconfig_init() {
// INIT-NEXT:    aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay, expand_mode = 0 : i32}
// INIT-NEXT:  }
// INIT:       aie.runtime_sequence(%arg0: memref<16xi32>)
// INIT-NOT:   aiex.npu.load_pdi
// INIT:       {{^    }}}

module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %compute = aie.tile(0, 2)
    aie.switchbox(%mem) {
      aie.connect<South : 0, TileControl : 0>
    }
    aie.switchbox(%compute) {
      %a = aie.amsel<0> (0)
      aie.masterset(TileControl : 0, %a)
      aie.packet_rules(South : 3) {
        aie.rule(31, 15, %a)
      }
    }
  }
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %compute = aie.tile(0, 2)
    aie.switchbox(%mem) {
      aie.connect<South : 0, TileControl : 0> {aie.is_ctrl_pkt_overlay}
    }
    aie.switchbox(%compute) {
      aie.connect<South : 1, DMA : 0>
      %a = aie.amsel<0> (0)
      %b = aie.amsel<1> (0)
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_pkt_overlay}
      aie.masterset(Core : 0, %b)
      aie.packet_rules(South : 3) {
        aie.rule(31, 15, %a) {aie.is_ctrl_pkt_overlay}
        aie.rule(31, 3, %b)
      }
    }
    %buf = aie.buffer(%mem) {address = 0 : i32, sym_name = "buf"} : memref<16xi32>
    %memdma = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.load_pdi {device_ref = @cfg, expand_mode = 2 : i32}
      aiex.npu.sync(%c0, %c0, %c0, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.npu.load_pdi {device_ref = @cfg, expand_mode = 2 : i32}
      aiex.npu.sync(%c0, %c0, %c0, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// A core is disabled, and only the channel its tile starts is reset.

// CHECK-LABEL: aie.runtime_sequence(%arg0: memref<16xi32>)
// CHECK:     %[[CORE:.*]] = arith.constant 2301952 : i32
// CHECK:     %[[ON:.*]] = arith.constant 1 : i32
// CHECK:     aiex.npu.maskwrite32(%[[CORE]], %[[ON]], %[[ON]])
// CHECK:     %[[TD_CORE:.*]] = arith.constant 2301952 : i32
// CHECK:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:     %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK:     aiex.npu.maskwrite32(%[[TD_CORE]], %[[ZERO]], %[[ENABLE]])
// CHECK:     %[[S2MM0:.*]] = arith.constant 2219520 : i32
// CHECK:     %[[RESET:.*]] = arith.constant 2 : i32
// CHECK:     aiex.npu.maskwrite32(%[[S2MM0]], %[[RESET]], %[[RESET]])
// CHECK:     aiex.npu.maskwrite32(%[[S2MM0]], %[[ZERO]], %[[RESET]])
// CHECK-NOT: aiex.npu
// CHECK:     {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %compute = aie.tile(0, 2)
    %buf = aie.buffer(%compute) {address = 1024 : i32, sym_name = "buf"} : memref<16xi32>
    %mem = aie.mem(%compute) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
    %core = aie.core(%compute) {
      aie.end
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// A load this pass leaves alone sets up a configuration no teardown undoes, so
// the load after it keeps its reset.

// CHECK-LABEL: aie.runtime_sequence(%arg0: memref<16xi32>)
// CHECK:     aiex.npu.load_pdi {device_ref = @empty_0
// CHECK:     aiex.npu.load_pdi {device_ref = @cfg, expand_mode = 0
// CHECK:     aiex.npu.load_pdi {device_ref = @empty_0
// CHECK-NOT: aiex.npu.load_pdi
// CHECK:     {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %compute = aie.tile(0, 2)
    aie.switchbox(%compute) {
      aie.connect<South : 1, DMA : 0>
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @cfg}
      aiex.npu.load_pdi {device_ref = @cfg, expand_mode = 0 : i32}
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// A device whose loads are of both kinds gets no init entry, since each
// sequence keeps its first preload.

// INIT-LABEL: aie.device(npu2_1col) @mixed
// INIT-NOT:   reconfig_init
// INIT:       aiex.npu.load_pdi {device_ref = @empty_0
// INIT:       aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay
// INIT-NOT:   reconfig_init

module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 0)
  }
  aie.device(npu2_1col) @mixed {
    %t = aie.tile(0, 0)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @mixed, expand_mode = 1 : i32}
      aiex.npu.load_pdi {device_ref = @mixed, expand_mode = 2 : i32}
    }
  }
}
