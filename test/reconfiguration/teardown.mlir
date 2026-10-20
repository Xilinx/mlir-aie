//===- teardown.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="teardown=true inline-config=true" --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-expand-load-pdi="teardown=true init-entry=true" --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=INIT
// RUN: not aie-opt --aie-expand-load-pdi="init-entry=true" --split-input-file %s 2>&1 | FileCheck %s --check-prefix=NO-TEARDOWN
// RUN: not aie-opt --aie-expand-load-pdi="ctrl-pkt=true teardown=true" --split-input-file %s 2>&1 | FileCheck %s --check-prefix=CTRL-PKT

// NO-TEARDOWN: error: aie-expand-load-pdi: init-entry requires teardown
// CTRL-PKT: error: teardown is not implemented for control packet reconfiguration

// @cfg sets a circuit and a packet route in tile (0, 2) and starts the
// memtile's MM2S channel 1. Its teardown, where the next configuration starts,
// zeroes the circuit's master and slave ports (2355204, 2355480), the packet
// route's (2355200, 2355484) and its slot (2355824), and pulses the reset bit
// of the memtile channel (1705528). Only the first configuration keeps its
// reset, and nothing is appended.

// CHECK-LABEL: aie.runtime_sequence(%arg0: memref<16xi32>)
// CHECK-DAG:  %[[A0:.*]] = arith.constant 2355204 : i32
// CHECK-DAG:  %[[A1:.*]] = arith.constant 2355480 : i32
// CHECK-DAG:  %[[A2:.*]] = arith.constant 2355200 : i32
// CHECK-DAG:  %[[A3:.*]] = arith.constant 2355484 : i32
// CHECK-DAG:  %[[A4:.*]] = arith.constant 2355824 : i32
// CHECK-DAG:  %[[RESET:.*]] = arith.constant 1705528 : i32
// CHECK-DAG:  %[[BIT:.*]] = arith.constant 2 : i32
// CHECK-DAG:  %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG:  %[[SYNC_ADDR:.*]] = arith.constant 119316 : i32
// CHECK:      aiex.npu.load_pdi {device_ref = @empty_0
// CHECK:      aiex.npu.write32(%[[SYNC_ADDR]], %{{.*}})
// CHECK:      aiex.npu.sync
// CHECK:      aiex.npu.write32(%[[SYNC_ADDR]], %{{.*}})
// CHECK-NEXT: aiex.npu.write32(%[[A0]], %[[ZERO]])
// CHECK-NEXT: aiex.npu.write32(%[[A1]], %[[ZERO]])
// CHECK-NEXT: aiex.npu.write32(%[[A2]], %[[ZERO]])
// CHECK-NEXT: aiex.npu.write32(%[[A3]], %[[ZERO]])
// CHECK-NEXT: aiex.npu.write32(%[[A4]], %[[ZERO]])
// CHECK-NEXT: aiex.npu.maskwrite32(%[[RESET]], %[[BIT]], %[[BIT]])
// CHECK-NEXT: aiex.npu.maskwrite32(%[[RESET]], %[[ZERO]], %[[BIT]])
// CHECK-NOT:  aiex.npu.load_pdi
// CHECK:      aiex.npu.sync
// CHECK-NEXT: aiex.npu.write32(%[[A0]], %[[ZERO]])
// CHECK:      aiex.npu.maskwrite32
// CHECK:      aiex.npu.maskwrite32
// CHECK-NOT:  aiex.npu.load_pdi
// CHECK:      {{^    }}}

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

// INIT:       aie.device(npu2) @empty_0
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

  aie.device(npu2) @main {
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

// A core is disabled, and only the channel its tile starts is reset.

// CHECK-LABEL: aie.runtime_sequence(%arg0: memref<16xi32>)
// CHECK-DAG:  %[[CORE:.*]] = arith.constant 2301952 : i32
// CHECK-DAG:  %[[ON:.*]] = arith.constant 1 : i32
// CHECK-DAG:  %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG:  %[[S2MM0:.*]] = arith.constant 2219520 : i32
// CHECK-DAG:  %[[RESET:.*]] = arith.constant 2 : i32
// CHECK:      aiex.npu.maskwrite32(%[[CORE]], %[[ON]], %[[ON]])
// CHECK-NEXT: aiex.npu.maskwrite32(%[[CORE]], %[[ZERO]], %[[ON]])
// CHECK-NEXT: aiex.npu.maskwrite32(%[[S2MM0]], %[[RESET]], %[[RESET]])
// CHECK-NEXT: aiex.npu.maskwrite32(%[[S2MM0]], %[[ZERO]], %[[RESET]])
// CHECK-NEXT: {{^    }}}

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

// @second ends with a configuration no teardown undoes, and the host may
// dispatch either sequence after it, so both keep their reset and the device
// gets no init entry.

// INIT-LABEL: aie.device(npu2_1col) @tail
// INIT-NOT:   reconfig_init
// INIT:       aie.runtime_sequence @first(%arg0: memref<16xi32>)
// INIT-NEXT:    aiex.npu.load_pdi {device_ref = @empty_0
// INIT:       aie.runtime_sequence @second(%arg0: memref<16xi32>)
// INIT-NEXT:    aiex.npu.load_pdi {device_ref = @empty_1
// INIT:         aiex.npu.load_pdi {device_ref = @tail, expand_mode = 0

module {
  aie.device(npu2_1col) @tail {
    %compute = aie.tile(0, 2)
    aie.switchbox(%compute) {
      aie.connect<South : 1, DMA : 0>
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence @first(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @tail}
    }
    aie.runtime_sequence @second(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @tail}
      aiex.npu.load_pdi {device_ref = @tail, expand_mode = 0 : i32}
    }
  }
}
