//===- teardown_shim_dma.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="teardown=true" --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %ext = aie.external_buffer {sym_name = "ext"} : memref<16xi32>
    %shimdma = aie.shim_dma(%shim) {
      // expected-error@+1 {{starts a shim DMA channel, which teardown cannot stop}}
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%ext : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %ext = aie.external_buffer {sym_name = "ext"} : memref<16xi32>
    %lock = aie.lock(%shim, 0) {init = 1 : i32}
    %shimdma = aie.shim_dma(%shim) {
      // expected-error@+1 {{starts a shim DMA channel, which teardown cannot stop}}
      %0 = aie.dma(MM2S, 0) [{
        %c1 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1)
        aie.dma_bd(%ext : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(%lock, Release, %c1)
      }]
      aie.end
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}
