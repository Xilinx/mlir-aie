//===- cpp_npu_cpp_sequence.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// The runtime sequence as a C++ TXN builder, requested on its own. @nd_dynamic
// takes its d2 size as a runtime argument, so no flat instruction binary can
// stand in for this artifact: the size only reaches the BD word at TXN-build
// time. Requesting it alone builds no overlay, which is the point -- one
// shape-agnostic overlay can drive many sequences without rebuilding an
// identical xclbin per sequence.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aiecc --get-npu-cpp --npu-cpp-name=%t.d/seq.cpp --output-dir=%t.d \
// RUN:   --tmpdir=%t.d --verbose %s 2>&1 | FileCheck %s --check-prefix=BUILD
// RUN: FileCheck %s --check-prefix=CPP < %t.d/seq.cpp

// The C++ builder is written, and the overlay branch never runs.
// BUILD: wrote edge 'seq.cpp'
// BUILD-NOT: aie.xclbin
// BUILD-NOT: wrote edge 'insts_

// The builder takes the runtime size as a parameter and assembles the stream.
// CPP: generate_txn_main_nd_dynamic
// CPP-SAME: int64_t
// CPP: std::vector<uint32_t>

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)

    aie.runtime_sequence @nd_dynamic(%in: memref<8192xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, %n, 8, 32][4096, 512, 32, 1]) {id = 0 : i64, metadata = @of_in} : memref<8192xi32>
    }
  }
}
