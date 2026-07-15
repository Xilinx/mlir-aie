//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Dynamic (runtime SSA size/stride) shim-NOC dma_memcpy_nd lowering: a static-
// template blockwrite plus per-word npu.write32 overrides computed from the
// runtime operands, with a host-side bounds guard for a runtime size landing in
// a narrow BD field.

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// A non-contiguous transfer with a runtime d1 size. d1 lands in the 10-bit
// wrap field, so a guard is emitted; the BD size/stride words become write32
// overrides after the template blockwrite. The override addresses pin the shim
// BD-word layout: buffer_length is word 0 (BD base 118784), d0 word 3 (+12),
// d1 word 4 (+16), d2 word 5 (+20), iteration word 6 (+24).
// CHECK-LABEL: @seq
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.assert_bd_field(%{{.*}}) {max = 1023 : i32}
// CHECK: aiex.npu.write32(%c118784{{.*}})
// CHECK: aiex.npu.write32(%c118796{{.*}})
// CHECK: aiex.npu.write32(%c118800{{.*}})
// CHECK: aiex.npu.write32(%c118804{{.*}})
// CHECK: aiex.npu.write32(%c118808{{.*}})
// CHECK: aiex.npu.address_patch
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @seq(%arg0: memref<4096xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][2, 4, %n, 32][2048, 256, 64, 1]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi32>
    }
  }
}

// -----

// A contiguous transfer with a runtime size takes linear mode: the count goes
// into buffer_length (word 0, full width) and no d0/d1 guard is needed. Only
// word 0 (buffer_length) and word 6 (iteration) are overridden.
// CHECK-LABEL: @lin
// CHECK: aiex.npu.blockwrite
// CHECK-NOT: aiex.npu.assert_bd_field
// CHECK: aiex.npu.write32
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @lin(%arg0: memref<8192xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %n, 32][0, 0, 32, 1]) {id = 0 : i64, metadata = @alloc0} : memref<8192xi32>
    }
  }
}
