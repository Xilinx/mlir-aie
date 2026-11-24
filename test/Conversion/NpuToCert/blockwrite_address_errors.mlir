// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A blockwrite whose buffer has no address is diagnosed once, whether it is
// lowered directly or split, rather than lowered at its word offset.

// RUN: aie-opt --aie-npu-to-cert --split-input-file --verify-diagnostics %s

aie.device(npu2) {
  %tile = aie.tile(1, 2)
  // expected-note@+1 {{This buffer must have an address.}}
  %buffer = aie.buffer(%tile) {sym_name = "buffer"} : memref<4097xi32>
  memref.global "private" constant @data : memref<4xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4xi32>
    // expected-error@+1 {{referenced buffer must have address assigned}}
    aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @buffer} : memref<4xi32>
  }
}

// -----

aie.device(npu2) {
  %tile = aie.tile(1, 2)
  // expected-note@+1 {{This buffer must have an address.}}
  %buffer = aie.buffer(%tile) {sym_name = "buffer"} : memref<4097xi32>
  memref.global "private" constant @data : memref<4096xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4096xi32>
    // expected-error@+1 {{referenced buffer must have address assigned}}
    aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @buffer} : memref<4096xi32>
  }
}
