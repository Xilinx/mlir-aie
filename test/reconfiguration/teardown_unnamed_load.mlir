//===- teardown_unnamed_load.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="teardown=true inline-config=true" %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @seq()
// CHECK:         aiex.npu.load_pdi {device_ref = @empty_
// CHECK:         aiex.npu.load_pdi {id = 4 : i32}
// CHECK:         aiex.npu.load_pdi {device_ref = @empty_

module {
  aie.device(npu2_1col) @cfg {
    %compute = aie.tile(0, 2)
    aie.switchbox(%compute) {
      aie.connect<South : 1, DMA : 0>
    }
  }
  aie.device(npu2_1col) @host {
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @cfg}
      aiex.npu.load_pdi {id = 4 : i32}
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}
