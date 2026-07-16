//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// -----

// A runtime innermost (d0) size on a sub-word element type is hardware-valid
// when its byte extent lands on a granule, so it is NOT rejected: the lowering
// emits a runtime realizability guard (value % 4 for int8 vs the 32-bit
// granule) that yields no stream host-side if the runtime value is unrealizable.
// CHECK-LABEL: @subgran
// CHECK: aiex.npu.assert_bd_divisible(%{{.*}}) {divisor = 4 : i32}
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @subgran(%arg0: memref<4096xi8>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 8, %n][0, 0, 8, 1]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi8>
    }
  }
}

// -----

// A runtime INNERMOST stride is supported (no compile-time constant-1 rule): the
// encoder resolves the d0 collapse with a select. For a granule-aligned element
// type (int32) no realizability guard is needed.
// CHECK-LABEL: @rt_inner_i32
// CHECK-NOT: aiex.npu.assert_bd_divisible
// CHECK: aiex.npu.blockwrite
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @rt_inner_i32(%arg0: memref<4096xi32>, %s: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 4, 8][0, 0, 16, %s]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi32>
    }
  }
}

// -----

// A runtime innermost stride on a sub-word type is guarded with the unit-stride
// exemption: stride 1 (contiguous) is realizable, a non-unit sub-granule stride
// is not, so the guard is `value == 1 || value % 4 == 0`.
// CHECK-LABEL: @rt_inner_i8
// CHECK: aiex.npu.assert_bd_divisible(%{{.*}}) {allow_unit, divisor = 4 : i32}
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @rt_inner_i8(%arg0: memref<4096xi8>, %s: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 4, 8][0, 0, 8, %s]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi8>
    }
  }
}
