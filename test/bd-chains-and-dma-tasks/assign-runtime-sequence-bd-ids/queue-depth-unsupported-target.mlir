//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --verify-diagnostics %s
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s 2>/dev/null | FileCheck %s

// Enforcement is on by default, but it needs a pollable occupancy register and
// not every target reports one: the AIE2PS shim relocates its DMA registers and
// its status block has never been confirmed against a spec, so the target model
// declines to guess an address a poll would then read garbage from.
//
// Where it cannot enforce, it warns and the build succeeds. Failing instead
// would reject designs that compile today on a target where nothing can be done
// about it, which is not a thing a default may do. What it must not do is go
// quiet -- so the note says enforcement was attempted and why it could not
// apply.

// CHECK-LABEL: @degrade
// CHECK-NOT:   aiex.npu.maskpoll
aie.device(xcve3858) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @degrade(%arg0: memref<1280xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 512 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 768 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t3)
    %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 1024 len = 256)
      aie.end
    }
    // expected-warning@+2 {{whose task queue is only 4 deep, with 4 push(es) not yet known to have completed}}
    // expected-note@+1 {{the compiler would have waited for a free slot here, but this target reports no pollable task-queue occupancy register}}
    aiex.dma_start_task(%t4)
  }
}
