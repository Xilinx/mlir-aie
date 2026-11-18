//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-dma-tasks-to-npu %s | FileCheck %s

// d0 pads are elements in the IR and 32-bit words in the BD, and the BD length
// counts the padding, as on the static path: 2 and 6 bf16 pad 1 and 3 words of
// four 16-element rows.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 32 : i32
// CHECK-SAME: d0_size = 4 : i32
// CHECK-SAME: d0_zero_after = 3 : i32, d0_zero_before = 1 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %buf = aie.buffer(%t) { address = 0 : i32 } : memref<64xbf16>
    aie.runtime_sequence(%arg0: memref<64xbf16>) {
      %t1 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%buf : memref<64xbf16> offset = 0 len = 64 sizes = [4, 8] strides = [8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 2, const_pad_after = 6>]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %buf = aie.buffer(%t) { address = 0 : i32 } : memref<64xbf16>
    aie.runtime_sequence(%arg0: memref<64xbf16>) {
      %t1 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-error@+1 {{Inner-most padding count must result in padding in 32-bit words.}}
        aie.dma_bd(%buf : memref<64xbf16> offset = 0 len = 56 sizes = [4, 8] strides = [8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 1, const_pad_after = 5>]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %buf = aie.buffer(%t) { address = 0 : i32 } : memref<64xbf16>
    aie.runtime_sequence(%arg0: memref<64xbf16>) {
      %t1 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-note@+2 {{Do not include the highest dimension size in transfer length}}
        // expected-error@+1 {{BD length is 64 bytes. Lowest three dimensions of data layout transformation would result in transfer of 128 bytes.}}
        aie.dma_bd(%buf : memref<64xbf16> offset = 0 len = 32 sizes = [4, 8] strides = [8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 2, const_pad_after = 6>]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
