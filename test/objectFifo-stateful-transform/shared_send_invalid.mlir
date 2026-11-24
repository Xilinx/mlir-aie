// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-objectfifo-split --verify-diagnostics %s

module @initialized_explicit {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst = aie.tile(0, 4)
    // expected-error @+1 {{push_queue does not support initialized objects}}
    aie.objectfifo @a(%src, {%dst}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>, dense<2> : memref<16xi32>]
  }
}

// -----

module @initialized_inferred {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst = aie.tile(0, 4)
    // expected-error @+1 {{push_queue does not support initialized objects}}
    aie.objectfifo @a(%src, {%dst}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>, dense<2> : memref<16xi32>]
    aie.objectfifo @b(%src, {%dst}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

module @memtile_explicit {
  aie.device(npu2) {
    %src = aie.tile(0, 1)
    %dst = aie.tile(0, 4)
    // expected-error @+1 {{push_queue requires a sending DMA on a core tile with core-issued task support}}
    aie.objectfifo @a(%src, {%dst}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

module @memtile_inferred {
  aie.device(npu2) {
    %src = aie.tile(0, 1)
    %dst = aie.tile(0, 4)
    // expected-error @+1 {{push_queue requires a sending DMA on a core tile with core-issued task support}}
    aie.objectfifo @a(%src, {%dst}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%dst}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

module @no_issuer_explicit {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst = aie.tile(0, 4)
    // expected-error @+1 {{push_queue requires producer releases in a local core to submit send tasks}}
    aie.objectfifo @a(%src, {%dst}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

module @no_issuer_inferred {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst = aie.tile(0, 4)
    // expected-error @+1 {{push_queue requires producer releases in a local core to submit send tasks}}
    aie.objectfifo @a(%src, {%dst}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%dst}, 2 : i32) {transport = #aie.transport<dma>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
  }
}
