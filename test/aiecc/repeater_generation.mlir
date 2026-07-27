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
// RUN: cat %t/ckpt/*/input_with_addresses.mlir | FileCheck --check-prefix=MLIR %s

// The routing failure is reported and a resumable checkpoint is written.
// CHECK: slave port packet rules exceed the 4-slot limit
// CHECK: aiecc: wrote checkpoint to
// CHECK: To reproduce, run: aiecc --resume={{.*}}/manifest.json

// The manifest records the resume argv (narrowed to the failed edge) and the
// captured frontier inputs.
// MANIFEST: "argv"
// MANIFEST: "--get=input_physical.mlir"
// MANIFEST: "frontier"
// MANIFEST: "input_with_addresses.mlir"

// The captured frontier IR is the pre-routing module holding the unroutable flow.
// MLIR: aie.packet_flow(20)

// based on test/create-packet-flows/subcube_cover_overbudget.mlir (IDs may differ)
aie.device(xcvc1902) {
  %11 = aie.tile(1, 1)
  aie.packet_flow(20) { aie.packet_source<%11, West : 0>  aie.packet_dest<%11, Core : 0> }
  aie.packet_flow(21) { aie.packet_source<%11, West : 0>  aie.packet_dest<%11, Core : 1> }
  aie.packet_flow(22) { aie.packet_source<%11, West : 0>  aie.packet_dest<%11, DMA : 0> }
  aie.packet_flow(23) { aie.packet_source<%11, West : 0>  aie.packet_dest<%11, DMA : 1> }
  aie.packet_flow(24) { aie.packet_source<%11, West : 0>  aie.packet_dest<%11, TileControl : 0> }
}
