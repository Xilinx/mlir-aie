//===- dma_endpoint_segments.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Omitting segments means [0], which is only unambiguous for a one-segment pool.
module @implicit_segment_on_partitioned_pool {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32}
      : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
      aie.objectfifo.segment @s1 {offset = 16 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{must list segments explicitly for a multi-segment pool}}
    aie.objectfifo.dma_endpoint @d(%tile) drains @p
  }
}

// -----

module @duplicate_segment {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32}
      : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
      aie.objectfifo.segment @s1 {offset = 16 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{segments must be named in increasing offset order}}
    aie.objectfifo.dma_endpoint @d(%tile) drains @p {segments = [@s0, @s0]}
  }
}

// -----

module @dimension_count {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32}
      : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
      aie.objectfifo.segment @s1 {offset = 16 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{dimensions has 1 entries for 2 selected segments}}
    aie.objectfifo.dma_endpoint @d(%tile) drains @p {
      segments = [@s0, @s1],
      dimensions = #aie<bd_dim_layout_array_array[[<size = 16, stride = 1>]]>
    }
  }
}

// -----

module @padding_count {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32}
      : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
      aie.objectfifo.segment @s1 {offset = 16 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{padDimensions has 1 entries for 2 selected segments}}
    aie.objectfifo.dma_endpoint @d(%tile) drains @p {
      segments = [@s0, @s1],
      padDimensions = #aie<bd_pad_layout_array_array[[<const_pad_before = 0, const_pad_after = 1>]]>
    }
  }
}

// -----

module @padding_rank {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32}
      : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{dimensions and padDimensions entry 0 have different ranks}}
    aie.objectfifo.dma_endpoint @d(%tile) drains @p {
      dimensions = #aie<bd_dim_layout_array_array[[<size = 4, stride = 4>, <size = 4, stride = 1>]]>,
      padDimensions = #aie<bd_pad_layout_array_array[[<const_pad_before = 0, const_pad_after = 1>]]>
    }
  }
}

// -----

module @dimension_bounds {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32}
      : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
      aie.objectfifo.segment @s1 {offset = 16 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{dimensions entry 1 exceeds selected segment 's1' of size 16}}
    aie.objectfifo.dma_endpoint @d(%tile) drains @p {
      segments = [@s0, @s1],
      dimensions = #aie<bd_dim_layout_array_array[
        [<size = 4, stride = 4>, <size = 4, stride = 1>],
        [<size = 5, stride = 4>, <size = 4, stride = 1>]]>
    }
  }
}

// -----

// Two selected segments carry two independent transforms, including an empty
// entry for a linear transfer.
module @valid {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32}
      : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
      aie.objectfifo.segment @s1 {offset = 16 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @d(%tile) drains @p {
      segments = [@s0, @s1],
      dimensions = #aie<bd_dim_layout_array_array[
        [<size = 4, stride = 4>, <size = 4, stride = 1>], []]>,
      padDimensions = #aie<bd_pad_layout_array_array[
        [<const_pad_before = 0, const_pad_after = 0>,
         <const_pad_before = 0, const_pad_after = 1>], []]>
    }
  }
}
