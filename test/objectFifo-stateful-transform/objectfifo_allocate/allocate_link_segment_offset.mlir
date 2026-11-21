//===- allocate_link_segment_offset.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split %s | FileCheck %s --check-prefix=SPLIT
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s --check-prefix=LOWER

// A join's input, or a distribute's output, whose segment starts past the
// pool object's first element, shares the pool in place as the leading one
// does: its core reaches the segment as a buffer of its own, aliasing each
// pool buffer at the segment's offset (alias_of, alias_offset), a plain memref
// a kernel's pointer argument takes. The allocator places each alias at its
// pool buffer's address plus the offset.

// The link tile's core writes the second half of each object in place; only
// the far tile's half, leading, arrives by DMA.

// SPLIT-LABEL: module @join_past_the_first
// SPLIT:       %[[T:.*]] = aie.tile(0, 2)
// SPLIT-NOT:   @in1__prod__dma
// SPLIT-DAG:   aie.objectfifo.core_endpoint @in1__prod(%[[T]]) fills @out__pool {segments = [@s1]}
// SPLIT-DAG:   aie.objectfifo.dma_endpoint @in0__cons__dma(%[[T]]) fills @out__pool {fifoName = "in0", segments = [@s0]}

// LOWER-LABEL: module @join_past_the_first
// LOWER:       %[[T:.*]] = aie.tile(0, 2)
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out__buf0"} : memref<32xi32>
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 1088 : i32, alias_of = @out__buf0, alias_offset = 64 : i32, sym_name = "out__buf0__at16"} : memref<16xi32>
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out__buf1"} : memref<32xi32>
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 16448 : i32, alias_of = @out__buf1, alias_offset = 64 : i32, sym_name = "out__buf1__at16"} : memref<16xi32>
// LOWER:       %[[F:.*]] = aie.tile(0, 5)
// LOWER:       aie.flow(%[[F]], DMA : 0, %[[T]], DMA : 0)
// LOWER-NOT:   aie.flow(%[[T]], DMA
// LOWER:       aie.core(%[[T]])
// LOWER:         func.call @fill(%{{.*}}) : (memref<16xi32>) -> ()
module @join_past_the_first {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %f = aie.tile(0, 5)
    aie.objectfifo @in0(%f, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 16] [])
    aie.objectfifo.allocate @in1 (%t)
    aie.objectfifo.allocate @out (%t)
    func.func private @fill(memref<16xi32>)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @in1(Produce, 1) : memref<16xi32>
      func.call @fill(%0) : (memref<16xi32>) -> ()
      aie.objectfifo.release @in1(Produce, 1)
      %1 = aie.objectfifo.acquire @out(Consume, 1) : memref<32xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
    aie.core(%f) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// A distribute's second and third segments read in place, by the link tile's
// core and by its north neighbour, each through its own alias of every pool
// buffer; the leading one leaves by DMA for a far tile.

// SPLIT-LABEL: module @distribute_past_the_first
// SPLIT-DAG:   %[[T:.*]] = aie.tile(0, 2)
// SPLIT-DAG:   %[[N:.*]] = aie.tile(0, 3)
// SPLIT-DAG:   aie.objectfifo.dma_endpoint @out0__prod__dma(%[[T]]) drains @in__cons__pool {fifoName = "out0", segments = [@s0]}
// SPLIT-DAG:   aie.objectfifo.core_endpoint @out1__cons(%[[T]]) drains @in__cons__pool {segments = [@s1]}
// SPLIT-DAG:   aie.objectfifo.core_endpoint @out2__cons(%[[N]]) drains @in__cons__pool {segments = [@s2]}
// SPLIT-NOT:   @out1__prod__dma
// SPLIT-NOT:   @out2__prod__dma

// LOWER-LABEL: module @distribute_past_the_first
// LOWER:       %[[T:.*]] = aie.tile(0, 2)
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "in__cons__buf0"} : memref<48xi32>
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 1088 : i32, alias_of = @in__cons__buf0, alias_offset = 64 : i32, sym_name = "in__cons__buf0__at16"} : memref<16xi32>
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 1152 : i32, alias_of = @in__cons__buf0, alias_offset = 128 : i32, sym_name = "in__cons__buf0__at32"} : memref<16xi32>
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "in__cons__buf1"} : memref<48xi32>
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 16448 : i32, alias_of = @in__cons__buf1, alias_offset = 64 : i32, sym_name = "in__cons__buf1__at16"} : memref<16xi32>
// LOWER-DAG:   aie.buffer(%[[T]]) {address = 16512 : i32, alias_of = @in__cons__buf1, alias_offset = 128 : i32, sym_name = "in__cons__buf1__at32"} : memref<16xi32>
// LOWER:       %[[N:.*]] = aie.tile(0, 3)
// LOWER:       %[[F:.*]] = aie.tile(0, 5)
// LOWER:       aie.flow(%[[T]], DMA : 0, %[[F]], DMA : 0)
// LOWER-NOT:   aie.flow
// LOWER:       aie.core(%[[T]])
// LOWER:         func.call @take(%{{.*}}) : (memref<16xi32>) -> ()
// LOWER:       aie.core(%[[N]])
// LOWER:         func.call @take(%{{.*}}) : (memref<16xi32>) -> ()
module @distribute_past_the_first {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %n = aie.tile(0, 3)
    %f = aie.tile(0, 5)
    aie.objectfifo @in(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @out0(%t, {%f}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2(%t, {%n}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out0, @out1, @out2] ([] [0, 16, 32])
    aie.objectfifo.allocate @in (%t)
    aie.objectfifo.allocate @out1 (%t)
    aie.objectfifo.allocate @out2 (%t)
    func.func private @take(memref<16xi32>)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @in(Produce, 1) : memref<48xi32>
      aie.objectfifo.release @in(Produce, 1)
      %1 = aie.objectfifo.acquire @out1(Consume, 1) : memref<16xi32>
      func.call @take(%1) : (memref<16xi32>) -> ()
      aie.objectfifo.release @out1(Consume, 1)
      aie.end
    }
    aie.core(%n) {
      %0 = aie.objectfifo.acquire @out2(Consume, 1) : memref<16xi32>
      func.call @take(%0) : (memref<16xi32>) -> ()
      aie.objectfifo.release @out2(Consume, 1)
      aie.end
    }
    aie.core(%f) {
      %0 = aie.objectfifo.acquire @out0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out0(Consume, 1)
      aie.end
    }
  }
}
