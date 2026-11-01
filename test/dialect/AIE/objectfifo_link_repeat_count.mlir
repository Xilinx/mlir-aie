//===- objectfifo_link_repeat_count.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Join inputs replicate independently at their producers.
module @join_unequal {
  aie.device(npu1) {
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    %link = aie.tile(1, 1)
    %out = aie.tile(1, 0)
    aie.objectfifo @a(%a, {%link}, 1 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%b, {%link}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%link, {%out}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 16] [])
  }
}

// -----

module @join_explicit_and_default {
  aie.device(npu1) {
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    %link = aie.tile(1, 1)
    %out = aie.tile(1, 0)
    aie.objectfifo @a(%a, {%link}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%b, {%link}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%link, {%out}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 16] [])
  }
}

// -----

module @join_default_and_explicit {
  aie.device(npu1) {
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    %link = aie.tile(1, 1)
    %out = aie.tile(1, 0)
    aie.objectfifo @a(%a, {%link}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%b, {%link}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%link, {%out}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 16] [])
  }
}

// -----

// Explicit one and absent both mean one transfer.
module @join_default_one {
  aie.device(npu1) {
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    %link = aie.tile(1, 1)
    %out = aie.tile(1, 0)
    aie.objectfifo @a(%a, {%link}, 1 : i32) {repeat_count = 1 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%b, {%link}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%link, {%out}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 16] [])
  }
}

// -----

module @join_defaults {
  aie.device(npu1) {
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    %link = aie.tile(1, 1)
    %out = aie.tile(1, 0)
    aie.objectfifo @a(%a, {%link}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%b, {%link}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%link, {%out}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 16] [])
  }
}

// -----

module @join_independent_output {
  aie.device(npu1) {
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    %link = aie.tile(1, 1)
    %out = aie.tile(1, 0)
    aie.objectfifo @a(%a, {%link}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%b, {%link}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%link, {%out}, 1 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 16] [])
  }
}

// -----

// Distribute outputs must agree, but need not match the input's repeat.
module @distribute_mismatched {
  aie.device(npu1) {
    %in = aie.tile(1, 0)
    %link = aie.tile(1, 1)
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    aie.objectfifo @in(%in, {%link}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%link, {%a}, 1 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%link, {%b}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{distribute outputs must have equal effective repeat counts: @a has 2, @b has 3}}
    aie.objectfifo.link [@in] -> [@a, @b] ([] [0, 16])
  }
}

// -----

module @distribute_default_mismatch {
  aie.device(npu1) {
    %in = aie.tile(1, 0)
    %link = aie.tile(1, 1)
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    aie.objectfifo @in(%in, {%link}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%link, {%a}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%link, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{distribute outputs must have equal effective repeat counts: @a has 3, @b has 1 (default)}}
    aie.objectfifo.link [@in] -> [@a, @b] ([] [0, 16])
  }
}

// -----

module @distribute_default_first_mismatch {
  aie.device(npu1) {
    %in = aie.tile(1, 0)
    %link = aie.tile(1, 1)
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    aie.objectfifo @in(%in, {%link}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%link, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%link, {%b}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{distribute outputs must have equal effective repeat counts: @a has 1 (default), @b has 3}}
    aie.objectfifo.link [@in] -> [@a, @b] ([] [0, 16])
  }
}

// -----

module @distribute_default_one {
  aie.device(npu1) {
    %in = aie.tile(1, 0)
    %link = aie.tile(1, 1)
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    aie.objectfifo @in(%in, {%link}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%link, {%a}, 1 : i32) {repeat_count = 1 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%link, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@a, @b] ([] [0, 16])
  }
}

// -----

module @distribute_defaults {
  aie.device(npu1) {
    %in = aie.tile(1, 0)
    %link = aie.tile(1, 1)
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    aie.objectfifo @in(%in, {%link}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%link, {%a}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%link, {%b}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@a, @b] ([] [0, 16])
  }
}

// -----

module @distribute_independent_input {
  aie.device(npu1) {
    %in = aie.tile(2, 2)
    %link = aie.tile(1, 1)
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 3)
    aie.objectfifo @in(%in, {%link}, 1 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%link, {%a}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%link, {%b}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@a, @b] ([] [0, 16])
  }
}
