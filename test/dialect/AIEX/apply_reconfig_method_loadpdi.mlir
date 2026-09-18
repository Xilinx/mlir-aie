//===- apply_reconfig_method_loadpdi.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// loadpdi method: a load_pdi is a full reset every time it is applied, so NO
// shared `init` is synthesized and each entrypoint KEEPS its own sole
// self-reset load_pdi.

// RUN: aie-opt --aie-apply-reconfig-method %s | FileCheck %s

// CHECK-NOT:  @init
// CHECK:      aie.runtime_sequence @config_1(%{{.*}}: memref<8xi32>)
// CHECK:        aiex.npu.load_pdi {id = 3 : i32}
// CHECK:      aie.runtime_sequence @config_2(%{{.*}}: memref<8xi32>)
// CHECK:        aiex.npu.load_pdi {id = 4 : i32}
module {
  aie.device(npu2) {
    aie.runtime_sequence @config_1(%s: memref<8xi32>) {
      aiex.npu.load_pdi {id = 3 : i32}
    }
    aie.runtime_sequence @config_2(%s: memref<8xi32>) {
      aiex.npu.load_pdi {id = 4 : i32}
    }
  } {aiex.entrypoint = {reconfig_method = "loadpdi"}}
}
