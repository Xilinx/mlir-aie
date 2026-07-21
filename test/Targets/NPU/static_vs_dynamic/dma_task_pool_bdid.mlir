//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static-vs-dynamic equivalence for a RUNTIME bd_id drawn from the dynamic
// free-list pool. @pool_dynamic draws its BD id at runtime via
// dma_bd_pool_pop (aie-lower-dynamic-bd-pool), so the whole BD is emitted as
// write32s at bdBase + bd_id*0x20 rather than a constant-address blockwrite.
// @pool_static bakes the same descriptor at a pinned bd_id = 0.
//
// A fresh pool pops id 0 first, so the runtime id equals the pinned id and the
// two program the SAME final BD registers -- proven by replaying both streams
// into a register map (-DDYN_STRUCTURAL) rather than a byte diff. golden-vs-
// static stays byte-exact.
//
// The two sequences take different pipelines (the static allocator rejects the
// pool sequence's unpinned BD; the pool pass is a no-op on the pinned one), so
// they are kept in two files: this one is the static oracle, the pooled variant
// is Inputs/dma_task_pool_bdid_dyn.mlir. Both funcs go into one comparator.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d

// Static oracle: assign BD ids, lower tasks + queue, dma-to-npu.
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
// RUN:   --aie-dma-to-npu %s -o %t.d/static.mlir
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false \
// RUN:   -aie-sequence-name=pool_static %t.d/static.mlir > %t.d/golden.hex
// RUN: aie-translate --aie-npu-to-cpp %t.d/static.mlir > %t.d/gen_static.h

// Dynamic pool variant: lower the pool, canonicalize, then the same lowerings.
// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu \
// RUN:   %S/Inputs/dma_task_pool_bdid_dyn.mlir -o %t.d/dynamic.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/dynamic.mlir > %t.d/gen_dynamic.h

// RUN: cat %t.d/gen_static.h %t.d/gen_dynamic.h > %t.d/gen.h
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen.h"' \
// RUN:   -DSTATIC_FN=generate_txn_main_pool_static \
// RUN:   -DDYN_FN=generate_txn_main_pool_dynamic -DARGVAL=0 \
// RUN:   %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/cmp.exe
// RUN: %t.d/cmp.exe %t.d/golden.hex

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @pool_static(%in: memref<1024xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%in : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
