//===- scratchpad_sync_after_load_pdi.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The parameter-sync preamble writes core buffers and sets a core lock, and
// loading a PDI resets both. The preamble therefore goes after the last
// top-level load_pdi of the sequence, or first when there is none.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-lower-scratchpad-parameters %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @after_pdis
// CHECK: aiex.npu.load_pdi {device_ref = @empty}
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @after_pdis_dev}
// CHECK-NEXT: aiex.npu.create_scratchpad {size = 4 : ui32}
// CHECK: aiex.npu.update_from_scratchpad
// CHECK: aiex.set_lock
module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2) @empty { }
  aie.device(npu2) @after_pdis_dev {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %v = aiex.read_scratchpad_parameter @n : i32
      aie.end
    }
    aie.runtime_sequence @after_pdis() {
      aiex.npu.load_pdi {device_ref = @empty}
      aiex.npu.load_pdi {device_ref = @after_pdis_dev}
    }
  }
}

// -----

// CHECK-LABEL: aie.runtime_sequence @no_pdi
// CHECK-NEXT: aiex.npu.create_scratchpad {size = 4 : ui32}
// CHECK: aiex.set_lock
// CHECK-NEXT: arith.constant 7 : i32
module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %v = aiex.read_scratchpad_parameter @n : i32
      aie.end
    }
    aie.runtime_sequence @no_pdi() {
      %c7 = arith.constant 7 : i32
    }
  }
}
