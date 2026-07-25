//===- checkpoint_resume_aiesim.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator

// The aiesim work-folder subgraph must round-trip through checkpoint/resume.
// A straight-through `--get-aiesim` build (which assembles the whole `sim/`
// folder from declarative graph edges) is compared against a build that is cut
// at an aiesim IR edge (`sim/flows_physical.mlir`), snapshotted with
// `--checkpoint`, and then completed with `--resume`. Resume reloads the routed
// flows from the checkpoint and rebuilds everything downstream, so its `sim/`
// descriptors must be byte-identical to the reference.

// RUN: rm -rf %t && mkdir -p %t

// Straight-through reference: assemble the whole sim/ folder in one go.
// RUN: %aiecc --get-aiesim --xchesscc --xbridge --tmpdir=%t/ref.prj %s

// Cut at the routed-flows IR edge and snapshot it. With --cut the build stops
// at the cut point, so only the prefix up to sim/flows_physical.mlir runs here.
// RUN: %aiecc --get-aiesim --xchesscc --xbridge --tmpdir=%t/cut.prj --cut='sim/flows_physical.mlir' --checkpoint=%t/cut.ckpt %s

// The captured frontier is textual MLIR, not a binary artifact.
// RUN: cat %t/cut.ckpt/*/flows_physical.mlir | FileCheck --check-prefix=IR %s
// IR: aie.device

// Resume rebuilds the graph from the checkpoint's argv, reloads the flows IR,
// and completes the sim/ folder in %t/cut.prj.
// RUN: %aiecc --resume=%t/cut.ckpt/manifest.json

// The resumed descriptors match the straight-through reference bit for bit.
// RUN: cmp %t/ref.prj/sim/reports/graph.xpe %t/cut.prj/sim/reports/graph.xpe
// RUN: cmp %t/ref.prj/sim/arch/aieshim_solution.aiesol %t/cut.prj/sim/arch/aieshim_solution.aiesol
// RUN: cmp %t/ref.prj/sim/config/scsim_config.json %t/cut.prj/sim/config/scsim_config.json
// RUN: cmp %t/ref.prj/sim/flows_physical.json %t/cut.prj/sim/flows_physical.json

module @checkpoint_resume_aiesim {
  aie.device(xcvc1902) {
    %tile13 = aie.tile(1, 3)

    %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
    %buf13_1 = aie.buffer(%tile13) { sym_name = "b" } : memref<256xi32>

    %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_lock" }
    %lock13_5 = aie.lock(%tile13, 5) { sym_name = "output_lock" }

    %core13 = aie.core(%tile13) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock13_3, "Acquire", %c1)
      aie.use_lock(%lock13_5, "Acquire", %c0)

      %idx = arith.constant 3 : index
      %val = memref.load %buf13_0[%idx] : memref<256xi32>
      %sum = arith.addi %val, %val : i32
      memref.store %sum, %buf13_1[%idx] : memref<256xi32>

      aie.use_lock(%lock13_3, "Release", %c0)
      aie.use_lock(%lock13_5, "Release", %c1)
      aie.end
    }
  }
}
