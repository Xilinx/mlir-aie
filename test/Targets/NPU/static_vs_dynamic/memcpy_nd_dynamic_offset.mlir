//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static-vs-dynamic equivalence for a runtime OFFSET on dma_memcpy_nd. A runtime
// offset flows into the address-patch arg_plus as arith; sizes/strides stay
// constant so the BD is a plain static blockwrite. Register-replay
// (-DDYN_STRUCTURAL) proves the runtime-offset stream programs the same
// registers (including the patched arg_plus) as the static-baked offset.
//
// @off_static bakes offset d3 = 64; @off_dynamic takes it as %n.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// RUN: aie-opt --aie-dma-to-npu %s -o %t.d/lowered.mlir

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=off_static %t.d/lowered.mlir > %t.d/golden.hex

// RUN: aie-translate --aie-npu-to-cpp %t.d/lowered.mlir > %t.d/gen.h

// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_off_static \
// RUN:   -DDYN_FN=generate_txn_main_off_dynamic -DARGVAL=64 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)

    aie.runtime_sequence @off_static(%in: memref<8192xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 64][1, 4, 8, 32][4096, 512, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }

    aie.runtime_sequence @off_dynamic(%in: memref<8192xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, %n][1, 4, 8, 32][4096, 512, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }
  }
}
