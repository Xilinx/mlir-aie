//===- repeater_generation.mlir --------------------------------*- MLIR -*-===//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verify that a routing (pathfinder) failure writes a resumable checkpoint
// reproducer: the failed edge's inputs plus a manifest recording the argv to
// replay just that edge via --resume.

// RUN: rm -rf %t && mkdir -p %t
// RUN: not %aiecc --get-core-elfs --enable-repeater-scripts --repeater-output-dir=%t/ckpt %s 2>&1 | FileCheck %s
// RUN: cat %t/ckpt/manifest.json | FileCheck --check-prefix=MANIFEST %s
// RUN: cat %t/ckpt/*/input_with_symbols.mlir | FileCheck --check-prefix=MLIR %s

// The routing failure is reported and a resumable checkpoint is written.
// CHECK: need 5 packet rules, and a slave port holds 4
// CHECK: aiecc: wrote checkpoint to
// CHECK: To reproduce, run: aiecc --resume={{.*}}/manifest.json

// The manifest records the resume argv (narrowed to the failed edge) and the
// captured frontier inputs.
// MANIFEST: "argv"
// MANIFEST: "--get=input_physical.mlir"
// MANIFEST: "frontier"
// MANIFEST: "input_with_symbols.mlir"

// The captured frontier IR is the pre-routing module holding the unroutable flow.
// MLIR: aie.packet_flow(20)

// based on test/create-packet-flows/subcube_cover_overbudget.mlir (IDs may differ)
aie.device(npu1_1col) {
  %01 = aie.tile(0, 1)
  %02 = aie.tile(0, 2)
  %sb01 = aie.switchbox(%01) {
    aie.connect<DMA : 0, South : 0>
    aie.connect<DMA : 1, South : 1>
    aie.connect<DMA : 2, South : 2>
    aie.connect<DMA : 3, South : 3>
  }
  %sb02 = aie.switchbox(%02) {
    aie.connect<DMA : 1, South : 1>
    aie.connect<Core : 0, South : 2>
    aie.connect<North : 0, South : 3>
  }
  aie.packet_flow(20) { aie.packet_source<%02, DMA : 0>  aie.packet_dest<%01, DMA : 0> }
  aie.packet_flow(21) { aie.packet_source<%02, DMA : 0>  aie.packet_dest<%01, DMA : 1> }
  aie.packet_flow(22) { aie.packet_source<%02, DMA : 0>  aie.packet_dest<%01, DMA : 2> }
  aie.packet_flow(23) { aie.packet_source<%02, DMA : 0>  aie.packet_dest<%01, DMA : 3> }
  aie.packet_flow(24) { aie.packet_source<%02, DMA : 0>  aie.packet_dest<%01, DMA : 4> }
}
