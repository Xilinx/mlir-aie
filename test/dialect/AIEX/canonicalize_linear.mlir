//===- canonicalize_linear.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Tests for NpuDmaMemcpyNdOp canonicalization: contiguous row-major access
// patterns are folded into the canonical linear form [s3,1,1,N][st3,0,0,1].
//
// This is the fix for github.com/Xilinx/mlir-aie/issues/2825.
//
// All tests use static literal sizes/strides so that:
//   (a) canonicalization sees constant values and can fire, and
//   (b) the pre-canonicalization op is in-bounds for the verifier.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize --split-input-file %s | FileCheck %s

// -----

// Basic 2D fold: sizes=[1,1,2,512] strides=[0,0,512,1]  ->
//                sizes=[1,1,1,1024] strides=[0,0,0,1]
//
// Motivating case from issue #2825: in production K can exceed 1023 (the d0
// wrap limit).  After folding, N is encoded in the wider linear-mode transfer
// length register, so no limit applies.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_2d
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_2d(%arg0 : memref<2x512xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 512][0, 0, 512, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<2x512xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// 3D fold: sizes=[1,3,4,5] strides=[0,20,5,1]  ->
//          sizes=[1,1,1,60] strides=[0,0,0,1]

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_3d
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 60][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_3d(%arg0 : memref<3x4x5xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 3, 4, 5][0, 20, 5, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<3x4x5xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Already in canonical linear form: the pattern must not fire (idempotent).

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @already_linear
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @already_linear(%arg0 : memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<4096xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Non-contiguous: stride1 (3) != size0 (4) — must NOT be folded.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @no_fold_strided
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 2, 4][0, 0, 3, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @no_fold_strided(%arg0 : memref<32xi32>) {
      // stride1=3 != size0=4: genuinely strided rows, cannot fold.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 4][0, 0, 3, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<32xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Repeat dimension (s3 > 1) is preserved through the fold.
// sizes=[2,1,2,4] strides=[4096,0,4,1]  ->  sizes=[2,1,1,8] strides=[4096,0,0,1]

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_with_repeat
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][2, 1, 1, 8][4096, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_with_repeat(%arg0 : memref<8192xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][2, 1, 2, 4][4096, 0, 4, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<8192xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// bf16 element type — motivating case from issue #2825.
// sizes=[1,1,2,512] strides=[0,0,512,1]  ->  sizes=[1,1,1,1024] strides=[0,0,0,1]
// In production K can be 1024+ (exceeding the d0 limit); the fold moves the
// total count into the wider linear-mode transfer-length register.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_bf16
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_bf16(%arg0 : memref<2x512xbf16>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 512][0, 0, 512, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<2x512xbf16>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Non-unit inner stride: stride0=2 means elements are not unit-stride.
// Must NOT be folded.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @no_fold_inner_stride
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 2, 4][0, 0, 4, 2]
module {
  aie.device(npu1) {
    aie.runtime_sequence @no_fold_inner_stride(%arg0 : memref<32xi32>) {
      // stride0=2: skips every other element, not a linear scan.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 4][0, 0, 4, 2])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<32xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Wrong stride2: size2 > 1 but stride2 != size0 * size1 — must NOT be folded.
// (stride1 is correct, only stride2 is wrong.)

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @no_fold_stride2
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 2, 3, 4][0, 7, 4, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @no_fold_stride2(%arg0 : memref<64xi32>) {
      // stride2=7 != size0*size1=4*3=12: non-contiguous outer loop, cannot fold.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 2, 3, 4][0, 7, 4, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<64xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Nonzero static offset in the innermost dimension is preserved unchanged.
// offsets[0] (d0) contributes offsets[0]*strides[0] = 4*1 = 4 to the linear
// offset, so the result is [0,0,0,4].

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_with_offset
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 4][1, 1, 1, 1024][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_with_offset(%arg0 : memref<2048xi32>) {
      // Offset of 4 elements in d0; sizes/strides fold as normal.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 4][1, 1, 2, 512][0, 0, 512, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<2048xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Regression: nonzero offset in a folded (non-innermost) dimension must be
// linearised into d0 rather than dropped.  This is the pattern from the
// packet_flow_fanout hardware test:
//   two slices of a 128x64 buffer, each 64x64, addressed via d1 offset.
//
// Op 2 has outermost-first offsets [0, 0, 64, 0], sizes [1, 1, 64, 64],
// strides [0, 0, 64, 1].  In innermost-first:
//   offsets = [0, 64, 0, 0], strides = [1, 64, 0, 0]
// linearOffset = 0*1 + 64*64 + 0*0 = 4096  -> new offsets [0, 0, 0, 4096].
//
// Before the fix the intermediate strides were zeroed without updating the
// offsets, silently mapping both slices to byte 0.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_d1_offset
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 4096][1, 1, 1, 4096][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_d1_offset(%arg0 : memref<128x64xi32>) {
      // Second 64x64 slice: outermost-first offset [0,0,64,0] selects row 64.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 64, 0][1, 1, 64, 64][0, 0, 64, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<128x64xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Regression: nonzero offset in d2 (the outermost non-repeat dimension) must
// also be linearised.  This is the pattern from ctrl_packet_reconfig_4x1_cores
// where four 64x64 tiles are sliced out of a 4x64x64 buffer using d2 offsets.
//
// shim_in_1: offsets [0, 1, 0, 0], sizes [1, 1, 64, 64], strides [0, 4096, 64, 1]
// In innermost-first: offsets=[0,0,1,0], strides=[1,64,4096,0]
// linearOffset = 0*1 + 0*64 + 1*4096 = 4096  -> new offsets [0, 0, 0, 4096].
//
// Before the fix strides[2]=4096 was zeroed without folding offsets[2]=1 into
// d0, so all four channels computed byte offset 0 (pointing to the same tile).

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_d2_offset
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 4096][1, 1, 1, 4096][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_d2_offset(%arg0 : memref<4x64x64xi8>) {
      // Second 64x64 tile (index 1 along the batch axis).
      aiex.npu.dma_memcpy_nd (%arg0[0, 1, 0, 0][1, 1, 64, 64][0, 4096, 64, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<4x64x64xi8>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// packet attribute is preserved after the fold.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_packet
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
// CHECK-SAME:        packet = <pkt_type = 0, pkt_id = 5>
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_packet(%arg0 : memref<2x512xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 512][0, 0, 512, 1],
                               packet = <pkt_id = 5, pkt_type = 0>)
        { metadata = @of_fromMem, id = 0 : i64 } : memref<2x512xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// issue_token attribute is preserved after the fold.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_issue_token
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
// CHECK-SAME:        issue_token = true
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_issue_token(%arg0 : memref<2x512xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 512][0, 0, 512, 1])
        { metadata = @of_fromMem, id = 0 : i64, issue_token = true } : memref<2x512xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// s1 == 1 with a nonzero st1.  The notation is outermost-first:
// [s3, s2, s1, s0][st3, st2, st1, st0], so s1 is the third element from the
// left.  When s1 == 1 the stride st1 is never applied, so any st1 value is
// semantically equivalent to zero.  The canonicalization should still fire.
//
// sizes=[1,2,1,4] strides=[0,4,99,1]  ->  sizes=[1,1,1,8] strides=[0,0,0,1]

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_size1_nonzero_stride1
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_size1_nonzero_stride1(%arg0 : memref<32xi32>) {
      // st1=99 (middle stride) with s1=1: the stride is never applied.
      // st2=4==s0=4 is the correct contiguous stride for s2=2.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 2, 1, 4][0, 4, 99, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<32xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// s2 (outermost non-repeat) == 1 with a nonzero st2.  The notation is
// outermost-first: [s3, s2, s1, s0][st3, st2, st1, st0], so s2 is the second
// element from the left.  When s2 == 1 the stride st2 is never applied.
//
// sizes=[1,1,2,4] strides=[0,99,4,1]  ->  sizes=[1,1,1,8] strides=[0,0,0,1]

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_size2_nonzero_stride2
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_size2_nonzero_stride2(%arg0 : memref<32xi32>) {
      // st2=99 (outermost non-repeat stride) with s2=1: the stride is never
      // applied.  st1=4==s0=4 is a valid contiguous stride for s1=2.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 4][0, 99, 4, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<32xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Both size1 and size2 == 1 with nonzero strides for both.
//
// sizes=[1,1,1,4] strides=[0,7,99,1]  ->  sizes=[1,1,1,4] strides=[0,0,0,1]

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_both_size1_size2_nonzero_strides
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_both_size1_size2_nonzero_strides(%arg0 : memref<4xi32>) {
      // st1=99 (middle stride) with s1=1 and st2=7 with s2=1: both strides
      // are never applied; the transfer is a single contiguous run of 4 elems.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 4][0, 7, 99, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<4xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Non-zero offset in the repeat (d3/outermost) dimension is passed through
// unchanged to the new op's outermost offset slot.
//
// offsets=[5, 0, 0, 0] sizes=[2, 1, 2, 4] strides=[4096, 0, 4, 1]
// (outermost-first)
// In innermost-first: offsets=[0,0,0,5], strides=[1,4,0,4096]
// linearOffset = 0*1 + 0*4 + 0*0 = 0  ->  new offsets [5, 0, 0, 0]
// sizes -> [2, 1, 1, 8], strides -> [4096, 0, 0, 1]

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_repeat_offset
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [5, 0, 0, 0][2, 1, 1, 8][4096, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_repeat_offset(%arg0 : memref<8192xi32>) {
      // Outermost offset of 5 repeat-strides; inner dims fold as normal.
      aiex.npu.dma_memcpy_nd (%arg0[5, 0, 0, 0][2, 1, 2, 4][4096, 0, 4, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<8192xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// s1 == 1 with a non-zero offset1 and a non-zero strides1.  Even though the
// loop over d1 runs only once (size==1), getOffsetInBytes() still multiplies
// offset * stride for every dimension, so the linearization must include
// offsets[1]*strides[1] regardless.
//
// outermost-first: offsets=[0, 0, 3, 0] sizes=[1, 2, 1, 4] strides=[0, 4, 99, 1]
// innermost-first: offsets=[0,3,0,0], strides=[1,99,4,0]
// linearOffset = 0*1 + 3*99 + 0*4 = 297 -> new offsets [0, 0, 0, 297]
// sizes -> [1, 1, 1, 8], strides -> [0, 0, 0, 1]

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_size1_nonzero_offset1
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 297][1, 1, 1, 8][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_size1_nonzero_offset1(%arg0 : memref<512xi32>) {
      // s1=1, so strides[1]=99 is never applied during the transfer, but
      // getOffsetInBytes() uses it to compute the start address.  The fold
      // must preserve offsets[1]*strides[1] = 3*99 = 297 in the d0 slot.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 3, 0][1, 2, 1, 4][0, 4, 99, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<512xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// Metadata and id attributes are preserved through the fold.
// Also verifies that a non-zero id value survives.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_preserves_metadata
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
// CHECK-SAME:        {id = 7 : i64, metadata = @my_alloc}
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_preserves_metadata(%arg0 : memref<2x512xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 512][0, 0, 512, 1])
        { metadata = @my_alloc, id = 7 : i64 } : memref<2x512xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @my_alloc (%tile, MM2S, 0)
  }
}

// -----

// burst_length is preserved unchanged through the fold.  burst_length is a
// ShimTile-only attribute; zero_before/after is a MemTile-only attribute.
// NpuDmaMemcpyNdOp canonicalization only fires on ShimTile ops (all
// aiex.npu.dma_memcpy_nd in a runtime_sequence target the shim level), so
// only burst_length needs to be tested here.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_preserves_burst_length
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
// CHECK-SAME:        burst_length = 64
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_preserves_burst_length(%arg0 : memref<2x512xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 512][0, 0, 512, 1])
        { metadata = @of_fromMem, id = 0 : i64,
          burst_length = 64 : i64 } : memref<2x512xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----

// 2D fold where both d0 and d1 exceed the 10-bit ND wrap limit (>1023):
// sizes=[1,1,1080,7680] strides=[0,0,7680,1] is contiguous and should fold
// to sizes=[1,1,1,8294400] strides=[0,0,0,1].
// Motivating case: 1920x1080 RGBA image (lineWidthInBytes=7680, height=1080).

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @fold_large_2d_image
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 8294400][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @fold_large_2d_image(%arg0 : memref<8294400xi8>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1080, 7680][0, 0, 7680, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<8294400xi8>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}
