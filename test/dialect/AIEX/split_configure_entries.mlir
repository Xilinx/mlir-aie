//===- split_configure_entries.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Explode IRON's single multi-`aiex.configure` runtime_sequence (on the
// `aiex.entry_device`-marked host device) into N per-config runtime_sequences,
// one `aiex.configure` each, in original program order. The referenced device
// symbol repeats (op0 appears twice), so the new sequence names carry the
// block-order index to disambiguate occurrences and preserve schedule order.
// Each new sequence keeps the original host block-argument signature; the
// subviews inside each configure re-reference the new sequence's own args.

// Three passes over the same output:
//   CHECK  -- order, per-occurrence naming, and per-sequence arg rebinding.
//   CFG    -- EXACTLY three aiex.configure total in the marked device
//             (proves one-per-sequence: no sequence smuggled a second configure,
//             and none was dropped).
//   SEQ    -- EXACTLY three (named) runtime_sequences and NO surviving original:
//             a dropped `seq.erase()` would leave the unnamed 3-configure
//             sequence behind (printed after the new ones), reddening SEQ-NOT.
// RUN: aie-opt --aie-split-configure-entries %s | FileCheck %s
// RUN: aie-opt --aie-split-configure-entries %s | FileCheck %s --check-prefix=CFG
// RUN: aie-opt --aie-split-configure-entries %s | FileCheck %s --check-prefix=SEQ

module {
  // Config templates (offset-agnostic bodies); the pass must not touch these.
  aie.device(npu1) @op0 {
    aie.runtime_sequence @sequence(%a: memref<8xi32>, %b: memref<8xi32>) {
    }
  }
  aie.device(npu1) @op1 {
    aie.runtime_sequence @sequence(%a: memref<8xi32>) {
    }
  }

  // The marked host device: ONE runtime_sequence holding three configures
  // (op0, op1, op0) becomes THREE runtime_sequences, one configure each.

  // Order + per-occurrence naming + per-sequence arg rebinding.
  // CHECK-LABEL: aie.device(npu1) {
  // CHECK:   aie.runtime_sequence @op0_0(%[[A0:.*]]: memref<16xi32>, %[[B0:.*]]: memref<16xi32>)
  // CHECK:     aiex.configure @op0 {
  // CHECK:       memref.subview %[[A0]]
  // CHECK:       memref.subview %[[B0]]
  // CHECK:       aiex.run @sequence
  // CHECK:   aie.runtime_sequence @op1_1(%[[A1:.*]]: memref<16xi32>, %{{.*}}: memref<16xi32>)
  // CHECK:     aiex.configure @op1 {
  // CHECK:       memref.subview %[[A1]]
  // CHECK:       aiex.run @sequence
  // CHECK:   aie.runtime_sequence @op0_2(%{{.*}}: memref<16xi32>, %[[B2:.*]]: memref<16xi32>)
  // CHECK:     aiex.configure @op0 {
  // CHECK:       memref.subview %[[B2]]
  // CHECK:       aiex.run @sequence

  // Exactly three configures in the marked device -- one per emitted sequence.
  // CFG-LABEL: aie.device(npu1) {
  // CFG-COUNT-3: aiex.configure @
  // CFG-NOT: aiex.configure

  // Exactly three named sequences and NO leftover unnamed original sequence.
  // SEQ-LABEL: aie.device(npu1) {
  // SEQ-COUNT-3: aie.runtime_sequence @
  // SEQ-NOT: aie.runtime_sequence
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.configure @op0 {
        %sv0 = memref.subview %arg0[0] [8] [1] : memref<16xi32> to memref<8xi32, strided<[1]>>
        %sv1 = memref.subview %arg1[0] [8] [1] : memref<16xi32> to memref<8xi32, strided<[1]>>
        aiex.run @sequence(%sv0, %sv1) : (memref<8xi32, strided<[1]>>, memref<8xi32, strided<[1]>>)
      }
      aiex.configure @op1 {
        %sv = memref.subview %arg0[0] [8] [1] : memref<16xi32> to memref<8xi32, strided<[1]>>
        aiex.run @sequence(%sv) : (memref<8xi32, strided<[1]>>)
      }
      aiex.configure @op0 {
        %sv0 = memref.subview %arg1[0] [8] [1] : memref<16xi32> to memref<8xi32, strided<[1]>>
        %sv1 = memref.subview %arg0[0] [8] [1] : memref<16xi32> to memref<8xi32, strided<[1]>>
        aiex.run @sequence(%sv0, %sv1) : (memref<8xi32, strided<[1]>>, memref<8xi32, strided<[1]>>)
      }
    }
  } {aiex.entry_device = {reconfig_method = "ctrlpkt"}}
}
