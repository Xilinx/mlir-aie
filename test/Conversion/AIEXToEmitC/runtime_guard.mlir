//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// End to end: a runtime d1 size on a shim dma_memcpy_nd is guarded with a
// cf.assert, which the C++ TXN builder lowers to an early refusal before the
// BD words are written. The range check is one unsigned compare, so both a
// zero size (wraps to UINT64_MAX after the subtract) and a size past the
// 10-bit wrap field are refused instead of truncated into the BD.

// RUN: aie-opt --aie-dma-to-npu %s | aie-translate --aie-npu-to-cpp | FileCheck %s

// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int64_t [[N:v[0-9]+]]) {
// CHECK:   uint64_t [[NU:v[0-9]+]] = (uint64_t) [[N]];
// CHECK:   uint64_t [[M1:v[0-9]+]] = [[NU]] - {{v[0-9]+}};
// CHECK:   int64_t [[M1S:v[0-9]+]] = (int64_t) [[M1]];
// CHECK:   uint64_t [[L:v[0-9]+]] = (uint64_t) [[M1S]];
// CHECK:   bool [[OK:v[0-9]+]] = [[L]] <= {{v[0-9]+}};
// CHECK:   if (!([[OK]])) return aie_runtime::txn_refused("a runtime DMA d1 size must be in [1:1023]");
// CHECK:   aie_runtime::txn_append_blockwrite(
// CHECK:   return aie_runtime::txn_refused("a runtime DMA access runs past the end of its 4096-element host buffer");
// CHECK:   aie_runtime::txn_append_arg_patch(
// CHECK:   return std::move(txn);
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, MM2S, 0)
    aie.runtime_sequence @seq(%arg0: memref<4096xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][2, 4, %n, 32][2048, 256, 64, 1]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi32>
    }
  }
}
