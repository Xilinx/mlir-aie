//===- split_multi_config_entry_invalid.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The pass fails loud on out-of-contract input rather than mis-splitting.

// RUN: aie-opt --aie-split-multi-config-entry --split-input-file --verify-diagnostics %s

// An unrecognized reconfig_method on the marker is rejected.
module {
  // expected-error @+1 {{unrecognized reconfig_method 'bogus'}}
  aie.device(npu2) {
    aie.runtime_sequence @config_1(%s: memref<8xi32>) {
      aiex.npu.load_pdi {id = 1 : i32}
    }
  } {aiex.entrypoint = {reconfig_method = "bogus"}}
}

// -----

// ctrlpkt requires the trailing ctrl-pkt-stream block arg (appended by
// aie-ctrl-packet-to-dma); a zero-arg entry means the pass ran out of order.
module {
  aie.device(npu2) {
    // expected-error @+1 {{has no block arguments}}
    aie.runtime_sequence @config_1() {
      aiex.npu.load_pdi {id = 1 : i32}
    }
  } {aiex.entrypoint = {reconfig_method = "ctrlpkt"}}
}
