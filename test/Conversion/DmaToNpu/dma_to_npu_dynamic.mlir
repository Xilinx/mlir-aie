//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic (runtime SSA size/stride) shim-NOC dma_memcpy_nd lowering: the whole
// BD register block is computed from the runtime operands and packed into one
// npu.blockwrite_values, with a host-side bounds guard for a runtime size
// landing in a narrow BD field.

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// A non-contiguous transfer with a runtime d1 size. d1 lands in the 10-bit
// wrap field, so a guard is emitted; the guards precede the block-write that
// consumes the guarded words. The block-write address is the BD register base
// (bd 0 on shim 0,0 = 118784) and it covers the word the address patch targets.
// CHECK-LABEL: @seq
// CHECK: aiex.npu.assert_bd_field(%{{.*}}) {max = 1023 : i32}
// CHECK: aiex.npu.blockwrite_values(%c118784{{.*}} : i32) values
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
// into buffer_length (word 0, full width) and no d0/d1 guard is needed.
// CHECK-LABEL: @lin
// CHECK-NOT: aiex.npu.assert_bd_field
// CHECK: aiex.npu.blockwrite_values
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
// CHECK: aiex.npu.blockwrite_values
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

// -----

// A runtime OFFSET is supported: the byte offset (offset * stride * elemBytes)
// is built with arith and flows through the SSA arg_plus of the address patch.
// Here offset %o with innermost stride 1 on i32 gives arg_plus = %o * 4.
// CHECK-LABEL: @rt_offset
// CHECK: %[[T:.*]] = arith.trunci %arg1 : i64 to i32
// CHECK: arith.muli %[[T]]
// CHECK: aiex.npu.address_patch(%{{.*}} : i32)
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @rt_offset(%arg0: memref<4096xi32>, %o: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, %o][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi32>
    }
  }
}

// -----

// A runtime offset paired with a runtime stride: offset * stride is a single
// arith.muli (both operands runtime). No made-up "constant stride" restriction.
// CHECK-LABEL: @rt_offset_stride
// CHECK: aiex.npu.address_patch(%{{.*}} : i32)
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @rt_offset_stride(%arg0: memref<4096xi32>, %o: i64, %st: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, %o, 0][1, 1, 4, 8][0, 0, %st, 1]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi32>
    }
  }
}

// -----

// A CONSTANT non-zero offset paired with a RUNTIME stride on the same dim: the
// offset*stride term isn't compile-time foldable, so arg_plus must be built
// with arith rather than via getOffsetInBytes() (which would read the runtime
// stride as a constant). Regression for that crash.
// CHECK-LABEL: @const_offset_rt_stride
// CHECK: aiex.npu.address_patch(%{{.*}} : i32)
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @const_offset_rt_stride(%arg0: memref<4096xi32>, %st: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 16, 0][1, 1, 4, 8][0, 0, %st, 1]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi32>
    }
  }
}

// -----

// A CONSTANT pure-repeat outer dimension (d3 size > 1, stride 0) paired with a
// runtime inner size: the zero d3 stride is the repeat case (carried by the
// queue push's repeat_count), which is legal exactly as on the static path.
// verifyConstBdRealizability must NOT reject the constant zero stride here (it
// only requires positive strides on d0..d2). This is the whole-array GEMM
// A/B-tile fetch pattern with a runtime tile size.
// CHECK-LABEL: @const_repeat_rt_size
// CHECK: aiex.npu.blockwrite_values
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @const_repeat_rt_size(%arg0: memref<8192xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][4, 1, %n, 32][0, 0, 32, 1]) {id = 0 : i64, metadata = @alloc0} : memref<8192xi32>
    }
  }
}
