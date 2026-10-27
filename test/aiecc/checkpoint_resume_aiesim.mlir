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
// at the IR edge feeding that subgraph (`perDevice_{0}.mlir`), snapshotted with
// `--checkpoint`, and then completed with `--resume`. Resume reloads the
// per-device IR from the checkpoint and rebuilds every `sim/` descriptor, so
// they must be byte-identical to the reference.

// RUN: rm -rf %t && mkdir -p %t

// Straight-through reference: assemble the whole sim/ folder in one go.
// RUN: %aiecc --get-aiesim --xchesscc --xbridge --tmpdir=%t/ref.prj %s

// Cut at the per-device IR edge and snapshot it. With --cut the build stops at
// the cut point, so only the prefix up to perDevice_{0}.mlir runs here.
// RUN: %aiecc --get-aiesim --xchesscc --xbridge --tmpdir=%t/cut.prj --cut='perDevice_{0}.mlir' --checkpoint=%t/cut.ckpt %s

// The captured frontier is textual MLIR, not a binary artifact.
// RUN: cat %t/cut.ckpt/*/*.mlir | FileCheck --check-prefix=IR %s
// IR: aie.device

// Resume rebuilds the graph from the checkpoint's argv, reloads the per-device
// IR, and completes the sim/ folder in %t/cut.prj.
// RUN: %aiecc --resume=%t/cut.ckpt/manifest.json

// The resumed descriptors match the straight-through reference bit for bit.
// RUN: cmp %t/ref.prj/sim/reports/graph.xpe %t/cut.prj/sim/reports/graph.xpe
// RUN: cmp %t/ref.prj/sim/arch/aieshim_solution.aiesol %t/cut.prj/sim/arch/aieshim_solution.aiesol
// RUN: cmp %t/ref.prj/sim/config/scsim_config.json %t/cut.prj/sim/config/scsim_config.json

module @checkpoint_resume_aiesim {
  aie.device(xcvc1902) {
    %tile13 = aie.tile(1, 3)

    aie.buffer @a(%tile13) : memref<256xi32>
    aie.buffer @b(%tile13) : memref<256xi32>

    aie.lock @input_lock(%tile13, 3)
    aie.lock @output_lock(%tile13, 5)

    %core13 = aie.core(%tile13) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aie.use_lock(@input_lock, "Acquire", %c1)
      aie.use_lock(@output_lock, "Acquire", %c0)

      %idx = arith.constant 3 : index
      %buf13_0 = aie.get_buffer @a : memref<256xi32>
      %val = memref.load %buf13_0[%idx] : memref<256xi32>
      %sum = arith.addi %val, %val : i32
      %buf13_1 = aie.get_buffer @b : memref<256xi32>
      memref.store %sum, %buf13_1[%idx] : memref<256xi32>

      aie.use_lock(@input_lock, "Release", %c0)
      aie.use_lock(@output_lock, "Release", %c1)
      aie.end
    }
  }
}
