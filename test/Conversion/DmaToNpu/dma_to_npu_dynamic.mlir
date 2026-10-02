//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic (runtime SSA size/stride) shim-NOC dma_memcpy_nd lowering: the whole
// BD register block is computed from the runtime operands and packed into one
// npu.blockwrite_values. Every runtime operand is guarded host-side with a
// cf.assert: its BD field range, the granule alignment, and the host buffer
// bounds of the whole walk. Guards that fold to true are dropped.

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// A non-contiguous transfer with a runtime d1 size. d1 lands in the 10-bit
// wrap field, so a guard is emitted; the guards precede the block-write that
// consumes the guarded words. The size bounds keep buffer_length in range, so
// it needs no guard. The block-write address is the BD register base (bd 0 on
// shim 0,0 = 118784) and it covers the word the address patch targets.
// CHECK-LABEL: @seq
// CHECK: %[[D1:.*]] = arith.subi %arg1, %c1{{.*}} : i64
// CHECK: %[[OK:.*]] = arith.cmpi ule, %[[D1]], %c1022{{.*}} : i64
// CHECK: cf.assert %[[OK]], "a runtime DMA d1 size must be in [1:1023]"
// CHECK-NOT: buffer_length
// CHECK: aiex.npu.blockwrite_values(%c118784{{.*}} : i32) values
// CHECK: cf.assert %{{.*}}, "a runtime DMA access runs past the end of its 4096-element host buffer"
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
// into buffer_length (word 0, full width), so only that field bounds it.
// CHECK-LABEL: @lin
// CHECK-NOT: [1:1023]
// CHECK: cf.assert %{{.*}}, "a runtime DMA d1 size must be in [1:4294967295]"
// CHECK: cf.assert %{{.*}}, "a runtime DMA transfer exceeds the 4294967295-granule BD buffer_length"
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
// CHECK: cf.assert %{{.*}}, "a runtime DMA d0 size must be in [1:4092]"
// CHECK: arith.remui %arg1, %c4{{.*}} : i64
// CHECK: cf.assert %{{.*}}, "a runtime DMA d0 size must be a multiple of 4 elements (whole 4-byte granules)"
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
// CHECK: cf.assert %{{.*}}, "a runtime DMA d0 stride must be in [1:1048576] when its size > 1"
// CHECK-NOT: multiple of
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
// CHECK: cf.assert %{{.*}}, "a runtime DMA d0 stride must be in [1:4194304] when its size > 1"
// CHECK: %[[UNIT:.*]] = arith.cmpi eq, %arg1, %c1{{.*}} : i64
// CHECK: %[[REM:.*]] = arith.remui %arg1, %c4{{.*}} : i64
// CHECK: %[[MUL:.*]] = arith.cmpi eq, %[[REM]], %c0{{.*}} : i64
// CHECK: %[[OK:.*]] = arith.ori %[[UNIT]], %[[MUL]] : i1
// CHECK: cf.assert %[[OK]], "a runtime DMA d0 stride must be a multiple of 4 elements (whole 4-byte granules)"
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
// is built in i64 and flows through the SSA arg_plus of the address patch.
// Here offset %o with innermost stride 1 on i32 gives arg_plus = %o * 4, which
// is always granule-aligned, so only the bounds guard (%o + 63 <= 4095) stays.
// CHECK-LABEL: @rt_offset
// CHECK: %[[END:.*]] = arith.addi %arg1, %c63{{.*}} : i64
// CHECK: arith.cmpi ule, %[[END]], %c4095{{.*}} : i64
// CHECK: cf.assert %{{.*}}, "a runtime DMA access runs past the end of its 4096-element host buffer"
// CHECK-NOT: aligned
// CHECK: %[[BYTES:.*]] = arith.muli %arg1, %c4{{.*}} : i64
// CHECK: aiex.npu.address_patch(%[[BYTES]] : i64)
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
// CHECK: arith.muli %arg1, %arg2 : i64
// CHECK: aiex.npu.address_patch(%{{.*}} : i64)
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
// CHECK: aiex.npu.address_patch(%{{.*}} : i64)
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

// -----

// A runtime pure repeat (constant zero outer stride) never touches the 6-bit
// iteration wrap, so it gets no iteration guard; the queue push refuses a
// count past its 8-bit repeat_count instead.
// CHECK-LABEL: @rt_repeat
// CHECK: cf.assert %{{.*}}, "a runtime DMA repeat count must be in [1:256]"
// CHECK-NOT: [1:64]
// CHECK: cf.assert %{{.*}}, "a runtime DMA repeat count exceeds the task queue's [0:255] range (at most 256 executions)"
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @rt_repeat(%arg0: memref<32xi32>, %r: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][%r, 1, 1, 32][0, 0, 0, 1]) {id = 0 : i64, metadata = @alloc0} : memref<32xi32>
    }
  }
}

// -----

// With a runtime outer stride the iteration wrap is written whenever that
// stride is positive, so the field it would land in is guarded.
// CHECK-LABEL: @rt_outer_stride
// CHECK: %[[PURE:.*]] = arith.cmpi eq, %arg2, %c0{{.*}} : i64
// CHECK: %[[IT:.*]] = arith.cmpi ule, %{{.*}}, %c63{{.*}} : i64
// CHECK: %[[OK:.*]] = arith.ori %[[PURE]], %[[IT]] : i1
// CHECK: cf.assert %[[OK]], "a runtime DMA iteration count must be in [1:64]"
// CHECK: cf.assert %{{.*}}, "a runtime DMA iteration stride must be in [0:1048576] when its size > 1"
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @rt_outer_stride(%arg0: memref<8192xi32>, %r: i64, %s: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][%r, 1, 1, 32][%s, 0, 0, 1]) {id = 0 : i64, metadata = @alloc0} : memref<8192xi32>
    }
  }
}

// -----

// A runtime d1 stride is checked in the element domain, before it is scaled
// to granules: one unsigned compare of stride-1 refuses 0 and anything that
// would wrap the 20-bit field (134217792 would otherwise encode as 64).
// CHECK-LABEL: @rt_d1_stride
// CHECK: %[[S1:.*]] = arith.subi %arg1, %c1{{.*}} : i64
// CHECK: %[[OK:.*]] = arith.cmpi ule, %[[S1]], %c1048575{{.*}} : i64
// CHECK: cf.assert %[[OK]], "a runtime DMA d1 stride must be in [1:1048576] when its size > 1"
// CHECK: cf.assert %{{.*}}, "a runtime DMA access runs past the end of its 8192-element host buffer"
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @rt_d1_stride(%arg0: memref<8192xi32>, %s: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 4, 32][0, 0, %s, 1]) {id = 0 : i64, metadata = @alloc0} : memref<8192xi32>
    }
  }
}

// -----

// A runtime d2 size has no wrap field of its own; it reaches the hardware
// through buffer_length, which is computed in 64 bits and guarded.
// CHECK-LABEL: @rt_d2_size
// CHECK: cf.assert %{{.*}}, "a runtime DMA d2 size must be in [1:4294967295]"
// CHECK: %[[LEN:.*]] = arith.muli %arg1, %c128{{.*}} : i64
// CHECK: %[[FITS:.*]] = arith.cmpi ule, %[[LEN]], %c4294967295{{.*}} : i64
// CHECK: cf.assert %[[FITS]], "a runtime DMA transfer exceeds the 4294967295-granule BD buffer_length"
// CHECK: cf.assert %{{.*}}, "a runtime DMA access runs past the end of its 8192-element host buffer"
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @rt_d2_size(%arg0: memref<8192xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, %n, 4, 32][0, 256, 32, 1]) {id = 0 : i64, metadata = @alloc0} : memref<8192xi32>
    }
  }
}
