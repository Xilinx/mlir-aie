//===- apply_reconfig_method.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ctrlpkt method: on the aiex.entry_device-marked device, synthesize ONE shared
// `init` (cloned from the first entrypoint's sole load_pdi, signature = the
// trailing ctrl-pkt-stream arg) inserted BEFORE the configs, and STRIP every
// per-config load_pdi re-arm (the in-band self-clear supplies each reset). The
// pass derives all of this from the marker's reconfig_method; it takes no
// options. Config-template devices (no marker) are untouched.

// RUN: aie-opt --aie-apply-reconfig-method %s | FileCheck %s

// The unmarked config-template device survives unchanged.
// CHECK:      aie.device(npu2) @cfg
// The synthesized standup carries the sole load_pdi (id 1, cloned from config_1)
// and its signature is the trailing ctrl-pkt-stream arg.
// CHECK:      aie.runtime_sequence @init(%{{.*}}: memref<8xi32>)
// CHECK:        aiex.npu.load_pdi {id = 1 : i32}
// The per-config entrypoints survive by name but their load_pdi re-arm is gone.
// CHECK:      aie.runtime_sequence @config_1(%{{.*}}: memref<8xi32>)
// CHECK-NOT:    aiex.npu.load_pdi
// CHECK:      aie.runtime_sequence @config_2(%{{.*}}: memref<8xi32>)
// CHECK-NOT:    aiex.npu.load_pdi
module {
  aie.device(npu2) @cfg {
    aie.runtime_sequence @s(%a: memref<8xi32>) {
    }
  }

  aie.device(npu2) {
    aie.runtime_sequence @config_1(%s: memref<8xi32>) {
      aiex.npu.load_pdi {id = 1 : i32}
    }
    aie.runtime_sequence @config_2(%s: memref<8xi32>) {
      aiex.npu.load_pdi {id = 2 : i32}
    }
  } {aiex.entry_device = {reconfig_method = "ctrlpkt"}}
}
