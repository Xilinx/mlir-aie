//===- resident_buffer_survive.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The TXN path checks that no reset comes between two configurations holding a
// buffer resident on the same global, whatever holds the array in between.

// RUN: not aie-translate --aie-npu-to-binary -aie-output-binary=false --aie-device-name=main --aie-sequence-name=abc %s 2>&1 | FileCheck %s --check-prefix=ABC
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false --aie-device-name=main --aie-sequence-name=same %s -o /dev/null
// RUN: not aie-translate --aie-npu-to-binary -aie-output-binary=false --aie-device-name=main --aie-sequence-name=wrap %s 2>&1 | FileCheck %s --check-prefix=WRAP
// RUN: not aie-translate --aie-npu-to-binary -aie-output-binary=false --aie-device-name=main --aie-sequence-name=written %s 2>&1 | FileCheck %s --check-prefix=WRITTEN
// RUN: not aie-translate --aie-npu-to-binary -aie-output-binary=false --aie-device-name=a --aie-sequence-name=own %s 2>&1 | FileCheck %s --check-prefix=OWN

// ABC: error: 'aiex.npu.load_pdi' op resets the array between @a and @c, which both hold a buffer resident on @w; needs --reconfig-teardown
// WRAP: error: 'aiex.npu.load_pdi' op resets the array between @c and @c, which both hold a buffer resident on @w when the sequence runs again; needs --reconfig-teardown
// WRITTEN: error: 'aiex.npu.load_pdi' op resets the array between @a and @c, which both hold a buffer resident on @w; needs --reconfig-teardown
// OWN: error: 'aiex.npu.load_pdi' op resets the array between @a and @c, which both hold a buffer resident on @w; needs --reconfig-teardown
module {
  memref.global "private" @w : memref<16xi32>

  aie.device(npu2) @a {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {sym_name = "bufa", resident = @w} : memref<16xi32>
    aie.runtime_sequence @own(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @b }
      aiex.npu.load_pdi { device_ref = @c }
    }
  }

  aie.device(npu2) @b {
    %tile = aie.tile(0, 2)
  }

  aie.device(npu2) @c {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {sym_name = "bufc", resident = @w} : memref<16xi32>
  }

  aie.device(npu2) @empty_0 {
  }

  aie.device(npu2) @empty_1 {
  }

  aie.device(npu2) @main {
    aie.runtime_sequence @abc(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @a }
      aiex.npu.load_pdi { device_ref = @b }
      aiex.npu.load_pdi { device_ref = @c }
    }
    aie.runtime_sequence @same(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @a }
      aiex.npu.load_pdi { device_ref = @a }
    }
    aie.runtime_sequence @wrap(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @c }
      aiex.npu.load_pdi { device_ref = @b }
    }
    aie.runtime_sequence @written(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @empty_0 }
      aiex.npu.write_config @a
      aiex.npu.load_pdi { device_ref = @empty_1 }
      aiex.npu.write_config @c
    }
  }
}
