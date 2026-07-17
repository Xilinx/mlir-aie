//===- repeater_generation.mlir --------------------------------*- MLIR -*-===//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verify that a routing (pathfinder) failure writes a resumable checkpoint
// reproducer: the failed edge's inputs plus a manifest recording the argv to
// replay just that edge via --resume.

// RUN: rm -rf %t && mkdir -p %t
// RUN: not %python aiecc.py --aie-generate-core-elfs --enable-repeater-scripts --repeater-output-dir=%t/ckpt %s 2>&1 | FileCheck %s
// RUN: cat %t/ckpt/manifest.json | FileCheck --check-prefix=MANIFEST %s
// RUN: cat %t/ckpt/*/input_with_addresses.mlir | FileCheck --check-prefix=MLIR %s

// The routing failure is reported and a resumable checkpoint is written.
// CHECK: 'aie.rule' op can lead to false packet id match for id 28, which is not supposed to pass through this port
// CHECK: aiecc: wrote checkpoint to
// CHECK: To reproduce, run: aiecc --resume={{.*}}/manifest.json

// The manifest records the resume argv (narrowed to the failed edge) and the
// captured frontier inputs.
// MANIFEST: "argv"
// MANIFEST: "--get=input_physical.mlir"
// MANIFEST: "frontier"
// MANIFEST: "input_with_addresses.mlir"

// The captured frontier IR is the pre-routing module holding the unroutable flow.
// MLIR: aie.packet_flow(28)

// from test/create-packet-flows/badpacket_flow.mlir
aie.device(npu1_1col) {
  %03 = aie.tile(0, 3)
  %02 = aie.tile(0, 2)
  %00 = aie.tile(0, 0)
  aie.packet_flow(28) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%02, TileControl : 0>
  }
  aie.packet_flow(29) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%03, TileControl : 0>
  }
  aie.packet_flow(26) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%03, DMA : 0>
  }
}
