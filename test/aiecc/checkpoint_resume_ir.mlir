//===- checkpoint_resume_ir.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// Checkpoint/resume must work at intermediate *IR* cut points, not just at
// binary artifacts: the saved frontier is a `.mlir` file that a resume
// re-parses into the shared MLIRContext and continues from. Two IR cuts are
// exercised -- a whole-module edge (input_physical.mlir) and a per-op fan-out
// edge (perCore_{0}.mlir, one module shared across many focus ops) -- and each
// resumed build's insts.bin must be byte-identical to a straight-through
// reference.

// RUN: rm -rf %t && mkdir -p %t

// Straight-through reference build.
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --npu-insts-name=%t/ref_insts.bin --xclbin-name=%t/ref.xclbin --tmpdir=%t/ref.prj %s

// Cut 1: the post-routing whole-module IR (a ModRef edge). The checkpoint holds
// textual MLIR, which resume re-parses before running everything downstream.
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --npu-insts-name=%t/phys_insts.bin --xclbin-name=%t/phys.xclbin --tmpdir=%t/phys.prj --get='input_physical.mlir' --checkpoint=%t/phys.ckpt %s
// RUN: cat %t/phys.ckpt/*/input_physical.mlir | FileCheck --check-prefix=IR %s
// RUN: rm -f %t/phys_insts.bin
// RUN: aiecc --resume=%t/phys.ckpt/manifest.json
// RUN: cmp %t/ref_insts.bin %t/phys_insts.bin

// Cut 2: per-core IR (an OpInModule fan-out edge). All items share one module
// and differ only in their focus op, so the checkpoint stores the module once
// (module.mlir) plus each item's focus-op index; resume parses it once and
// rebinds per item.
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --npu-insts-name=%t/core_insts.bin --xclbin-name=%t/core.xclbin --tmpdir=%t/core.prj --get='perCore_{0}.mlir' --checkpoint=%t/core.ckpt %s
// RUN: cat %t/core.ckpt/*/module.mlir | FileCheck --check-prefix=IR %s
// RUN: rm -f %t/core_insts.bin
// RUN: aiecc --resume=%t/core.ckpt/manifest.json
// RUN: cmp %t/ref_insts.bin %t/core_insts.bin

// The captured frontier is textual MLIR, not a binary artifact.
// IR: aie.device
// IR: aie.core

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @data(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index

      %subview = aie.objectfifo.acquire @data(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
      %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

      scf.for %i = %c0 to %c64 step %c1 {
        %val = memref.load %elem[%i] : memref<64xi32>
        memref.store %val, %elem[%i] : memref<64xi32>
      }

      aie.objectfifo.release @data(Consume, 1)
      aie.end
    }

    aie.runtime_sequence(%buf : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%buf[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @data}
    }
  }
}
