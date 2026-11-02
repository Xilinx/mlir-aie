//===- objectfifo_bad_time_link.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{a time link takes turns, so it needs several inputs or several outputs}}
    aie.objectfifo.link [@in] -> [@out] ([] []) {mode = #aie.link_mode<time>}
  }
}

// -----

module {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @x(%a, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%b, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p(%m, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q(%m, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{a time link merges or dispatches, not both}}
    aie.objectfifo.link [@x, @y] -> [@p, @q] ([] []) {mode = #aie.link_mode<time>}
  }
}

// -----

module {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @x(%a, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%b, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%m, {%a}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    // expected-error@+1 {{a time link moves whole objects and takes no offsets}}
    aie.objectfifo.link [@x, @y] -> [@out] ([0, 16] []) {mode = #aie.link_mode<time>}
  }
}

// -----

module {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @x(%a, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%b, {%m}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out(%m, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{a time link's participants carry one object type, but 'y' carries '!aie.objectfifo<memref<8xi32>>' and 'x' carries '!aie.objectfifo<memref<16xi32>>'}}
    aie.objectfifo.link [@x, @y] -> [@out] ([] []) {mode = #aie.link_mode<time>}
  }
}

// -----

module {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_a(%m, {%a, %b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_b(%m, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{each turn of a dispatch reaches one consumer, but 'to_a' has 2}}
    aie.objectfifo.link [@in] -> [@to_a, @to_b] ([] []) {mode = #aie.link_mode<time>}
  }
}

// -----

module {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @x(%a, {%m}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 1>>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%b, {%m}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 2>>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%m, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{a merge is one route with one header, but its inputs pin different ones}}
    aie.objectfifo.link [@x, @y] -> [@out] ([] []) {mode = #aie.link_mode<time>}
  }
}

// -----

module {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo @x(%a, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%b, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{a time link holds its objects on the link point, which a shim tile has no memory for}}
    aie.objectfifo.link [@x, @y] -> [@out] ([] []) {mode = #aie.link_mode<time>}
  }
}
