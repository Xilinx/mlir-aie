//===- write_teardown.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The next dispatch's load of @empty_0 is skipped, but every configuration
// since is torn down.
module {
  aie.device(npu2_1col) @empty_0 {
  }
  aie.device(npu2_1col) @a {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @b {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @empty_0}
      aiex.npu.write_config @a
      aiex.npu.write_teardown @a
      aiex.npu.write_config @b
      aiex.npu.write_teardown @b
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @a {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.write_config @a
      aiex.npu.write_teardown @a
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @empty_0 {
  }
  aie.device(npu2_1col) @a {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @b {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      // expected-note @+1 {{the PDI is loaded here}}
      aiex.npu.load_pdi {device_ref = @empty_0}
      aiex.npu.write_config @a
      aiex.npu.write_teardown @a
      aiex.npu.write_config @b
      aiex.npu.load_pdi {device_ref = @empty_0}
      // expected-error @+1 {{follows a load_pdi the firmware skips}}
      aiex.npu.write_config @a
      aiex.npu.write_teardown @a
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @a {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      // expected-error @+1 {{must follow the write_config of @a, with no load_pdi between}}
      aiex.npu.write_teardown @a
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @empty_0 {
  }
  aie.device(npu2_1col) @a {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @b {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @empty_0}
      aiex.npu.write_config @a
      // expected-error @+1 {{must follow the write_config of @b, with no load_pdi between}}
      aiex.npu.write_teardown @b
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @empty_0 {
  }
  aie.device(npu2_1col) @empty_1 {
  }
  aie.device(npu2_1col) @a {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @empty_0}
      aiex.npu.write_config @a
      aiex.npu.load_pdi {device_ref = @empty_1}
      // expected-error @+1 {{must follow the write_config of @a, with no load_pdi between}}
      aiex.npu.write_teardown @a
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @empty_0 {
  }
  aie.device(npu2_1col) @shim {
    %shim = aie.tile(0, 0)
    %ext = aie.external_buffer {sym_name = "ext"} : memref<16xi32>
    %shimdma = aie.shim_dma(%shim) {
      // expected-note @+1 {{started here}}
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%ext : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @empty_0}
      aiex.npu.write_config @shim
      // expected-error @+1 {{@shim starts a shim DMA channel, which teardown cannot stop}}
      aiex.npu.write_teardown @shim
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @empty_0 {
  }
  aie.device(npu2_1col) @shim {
    %shim = aie.tile(0, 0)
    %ext = aie.external_buffer {sym_name = "ext"} : memref<16xi32>
    %lock = aie.lock(%shim, 0) {init = 1 : i32}
    %shimdma = aie.shim_dma(%shim) {
      // expected-note @+1 {{started here}}
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
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @empty_0}
      aiex.npu.write_config @shim
      // expected-error @+1 {{@shim starts a shim DMA channel, which teardown cannot stop}}
      aiex.npu.write_teardown @shim
    }
  }
}
