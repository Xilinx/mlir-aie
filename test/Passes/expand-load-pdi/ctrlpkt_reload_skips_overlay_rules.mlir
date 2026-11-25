//===- ctrlpkt_reload_skips_overlay_rules.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: split-file %s %t
// RUN: aie-opt --aie-import-overlay --aie-expand-load-pdi="ctrl-pkt=true" %t/overlay.mlir | FileCheck %s
// RUN: %aiecc -j1 --load-pdi-to-ctrl-pkt --reconfig-teardown=false --get=npu_expanded.mlir --output-dir=%t/plain --tmpdir=%t/plain/work %t/constant.mlir
// RUN: FileCheck %s --check-prefix=CONSTANT --implicit-check-not=scf. < %t/plain/npu_expanded.mlir
// RUN: %aiecc -j1 --load-pdi-to-ctrl-pkt --get-full-elf --reconfig-teardown --get=npu_expanded.mlir --cut=npu_expanded.mlir --checkpoint=%t/teardown-checkpoint --output-dir=%t/teardown --tmpdir=%t/teardown/work %t/constant.mlir
// RUN: FileCheck %s --check-prefix=CONSTANT --implicit-check-not=scf. < %t/teardown-checkpoint/npu_expanded.mlir/npu_expanded.mlir
// RUN: not %aiecc -j1 --load-pdi-to-ctrl-pkt --get-full-elf --reconfig-teardown --get=npu_expanded.mlir --cut=npu_expanded.mlir --checkpoint=%t/dynamic-checkpoint --output-dir=%t/dynamic --tmpdir=%t/dynamic/work %t/dynamic.mlir 2>&1 | FileCheck %s --check-prefix=DYNAMIC

//--- overlay.mlir

// South:1 of tile (0,2) carries the overlay's rule in slot 0 and the design's
// in slot 1. A control-packet reload writes slot 1 alone: rewriting slot 0 or
// the slave port's configuration would drop the control packets the reload
// itself travels by.

// CHECK:      aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay
// CHECK-NEXT: aiex.set_device @design
// CHECK-NEXT: aiex.control_packet_barrier {independent}
// CHECK-NEXT:   aiex.control_packet {address = 2355204 : ui32
// CHECK-NEXT:   aiex.control_packet {address = 2355812 : ui32
// CHECK-NEXT: aiex.control_packet_barrier {independent}
// CHECK-NEXT: aiex.control_packet_barrier
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay_copy
// CHECK-NEXT: {{^}}    }

module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %shim = aie.tile(0, 0)
    aie.shim_mux(%shim) {
      aie.connect<DMA : 0, North : 3> {aie.is_ctrl_pkt_overlay}
    }
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {aie.is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu2_1col) @design {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      %b = aie.amsel<0> (0)
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_pkt_overlay}
      aie.masterset(DMA : 0, %b)
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a) {aie.is_ctrl_pkt_overlay}
        aie.rule(31, 2, %b)
      }
    }
  } {overlay = @ctrl_pkt_overlay}
  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design}
    }
  }
}

//--- constant.mlir

// CONSTANT-LABEL: aie.runtime_sequence
// CONSTANT: aiex.set_device @first
// CONSTANT: aiex.set_device @second
// CONSTANT: aiex.set_device @first
// CONSTANT: aiex.set_device @second
// CONSTANT-NOT: aiex.set_device
module {
  aie.device(npu2_1col) @first {
    %tile = aie.tile(0, 2)
    %lock = aie.lock(%tile, 0) {init = 0 : i32}
  }
  aie.device(npu2_1col) @second {
    %tile = aie.tile(0, 2)
    %lock = aie.lock(%tile, 0) {init = 1 : i32}
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @sequence() {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %condition = arith.constant true
      scf.for %iteration = %zero to %two step %one {
        scf.if %condition {
          aiex.npu.load_pdi {device_ref = @first}
        }
        aiex.npu.load_pdi {device_ref = @second}
      }
    }
  }
}

//--- dynamic.mlir

// DYNAMIC: with teardown, must be in the runtime sequence's body, not inside scf.for
module {
  aie.device(npu2_1col) @design {
    %tile = aie.tile(0, 2)
    %lock = aie.lock(%tile, 0) {init = 0 : i32}
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @sequence(%iterations: index) {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %iteration = %zero to %iterations step %one {
        aiex.npu.load_pdi {device_ref = @design}
      }
    }
  }
}
