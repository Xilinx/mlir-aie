//===- configure_once.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="configure-once=true" --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: not aie-opt --aie-expand-load-pdi="ctrl-pkt=true configure-once=true" --split-input-file %s 2>&1 | FileCheck %s --check-prefix=CTRLPKT
// CTRLPKT: configure-once does not apply to ctrl-pkt expansion

// A sequence that loads one device keeps its load: the firmware skips
// reloading the PDI it loaded last, so later runs do not reconfigure.
module {
  // CHECK-NOT: @empty_
  aie.device(npu2_1col) @dev_a {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      aie.connect<South : 0, Core : 0>
    }
  }

  aie.device(npu2_1col) @main {
    // CHECK-LABEL: aie.runtime_sequence(%arg0: memref<1xi32>)
    aie.runtime_sequence (%arg0: memref<1xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @dev_a, expand_mode = 0 : i32}
      // CHECK-NOT: aiex.npu.write32
      // CHECK-NOT: aiex.npu.load_pdi
      aiex.npu.load_pdi { device_ref = @dev_a }
    }
  }
}

// -----

// A sequence that switches between devices is still expanded.
module {
  aie.device(npu2_1col) @dev_a {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      aie.connect<South : 0, Core : 0>
    }
  }

  aie.device(npu2_1col) @dev_b {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      aie.connect<North : 0, Core : 0>
    }
  }

  aie.device(npu2_1col) @main {
    // CHECK-LABEL: aie.runtime_sequence(%arg0: memref<1xi32>)
    aie.runtime_sequence (%arg0: memref<1xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
      // CHECK: aiex.npu.write32
      // CHECK: aiex.npu.load_pdi {device_ref = @empty_1, expand_mode = 0 : i32}
      // CHECK: aiex.npu.write32
      aiex.npu.load_pdi { device_ref = @dev_a }
      aiex.npu.load_pdi { device_ref = @dev_b }
    }
  }
}

// -----

// An explicit expand_mode is kept.
module {
  aie.device(npu2_1col) @dev_a {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      aie.connect<South : 0, Core : 0>
    }
  }

  aie.device(npu2_1col) @main {
    // CHECK-LABEL: aie.runtime_sequence(%arg0: memref<1xi32>)
    aie.runtime_sequence (%arg0: memref<1xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
      // CHECK: aiex.npu.write32
      aiex.npu.load_pdi { device_ref = @dev_a, expand_mode = 1 : i32 }
    }
  }
}

// -----

// A reference to a symbol that is not a device is still reported.
module {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence (%arg0: memref<1xi32>) {
      // expected-error @+1 {{Referenced symbol 'missing' is not a device}}
      aiex.npu.load_pdi { device_ref = @missing }
    }
  }
}
