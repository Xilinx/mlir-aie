//===- bad_write_config.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// npu.write_config writes over whatever the array holds, so it must follow a
// reset the firmware runs: a load of an empty device's PDI other than the one
// it loaded last. Ahead of a sequence's first load that is its last load, as
// the host runs the sequence again on each dispatch.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2_1col) @init {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      // expected-error @+1 {{must follow the load_pdi that resets the array}}
      aiex.npu.write_config @init
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @init {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @init}
      // expected-error @+1 {{must follow a load_pdi of an empty device}}
      aiex.npu.write_config @init
      aiex.npu.load_pdi {device_ref = @main}
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @empty_0 {
  }
  aie.device(npu2_1col) @empty_1 {
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @empty_0}
      // expected-error @+1 {{@missing is not a device}}
      aiex.npu.write_config @missing
      aiex.npu.load_pdi {device_ref = @empty_1}
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @empty_0 {
  }
  aie.device(npu2_1col) @empty_1 {
  }
  aie.device(npu2_1col) @init {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      // expected-note @+1 {{the PDI is loaded here}}
      aiex.npu.load_pdi {device_ref = @empty_0}
      aiex.npu.write_config @init
      aiex.npu.load_pdi {device_ref = @empty_0}
      // expected-error @+1 {{follows a load_pdi the firmware skips}}
      aiex.npu.write_config @init
      aiex.npu.load_pdi {device_ref = @empty_1}
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @empty_0 {
  }
  aie.device(npu2_1col) @init {
    %t = aie.tile(0, 2)
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      // expected-note @+1 {{the PDI is loaded here}}
      aiex.npu.load_pdi {device_ref = @empty_0}
      // expected-error @+1 {{follows a load_pdi the firmware skips}}
      aiex.npu.write_config @init
    }
  }
}
