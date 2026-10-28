// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s
// RUN: aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=false' --verify-diagnostics %s

// Neither occupancy analysis alone nor default-on completion tracking should
// distinguish dead branch-local task identities at joins or loop backedges.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @local_owners(%buf: memref<256xi32>, %n: index, %cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      scf.if %cond {
        %a = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        }
        aiex.dma_start_task(%a)
      }
      scf.if %cond {
        %b = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        }
        aiex.dma_start_task(%b)
      }
      scf.if %cond {
        %c = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        }
        aiex.dma_start_task(%c)
      }
      scf.if %cond {
        %d = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        }
        aiex.dma_start_task(%d)
      }
      scf.if %cond {
        %e = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        }
        aiex.dma_start_task(%e)
      }
      scf.if %cond {
        %f = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        }
        aiex.dma_start_task(%f)
      }
      scf.if %cond {
        %g = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        }
        aiex.dma_start_task(%g)
      }
      scf.if %cond {
        %h = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        }
        aiex.dma_start_task(%h)
      }
    }
  }
}
