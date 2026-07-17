//===- checkpoint_resume.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// Checkpoint/resume must produce identical outputs. A straight-through xclbin +
// insts.bin build is compared against builds that are split into a
// `--checkpoint` at a cut point followed by a `--resume`, for cut points at
// increasing depth: per-core objects (early), linked ELFs (core compilation),
// and the PDI (just before final xclbin assembly). Each cut's resumed insts.bin
// must be byte-identical to the reference.

// RUN: rm -rf %t && mkdir -p %t

// Straight-through reference build.
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --npu-insts-name=%t/ref_insts.bin --xclbin-name=%t/ref.xclbin --tmpdir=%t/ref.prj %s

// Cut 1: per-core objects. Checkpoint at the object compile edge, drop the
// produced insts, then resume to completion and compare.
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --npu-insts-name=%t/obj_insts.bin --xclbin-name=%t/obj.xclbin --tmpdir=%t/obj.prj --cut='objects_{0}.o' --checkpoint=%t/obj.ckpt %s
// RUN: rm -f %t/obj_insts.bin
// RUN: aiecc --resume=%t/obj.ckpt/manifest.json
// RUN: cmp %t/ref_insts.bin %t/obj_insts.bin

// Cut 2: linked core ELFs.
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --npu-insts-name=%t/elf_insts.bin --xclbin-name=%t/elf.xclbin --tmpdir=%t/elf.prj --cut='elfs_{0}.elf' --checkpoint=%t/elf.ckpt %s
// RUN: rm -f %t/elf_insts.bin
// RUN: aiecc --resume=%t/elf.ckpt/manifest.json
// RUN: cmp %t/ref_insts.bin %t/elf_insts.bin

// Cut 3: PDI (immediately before xclbin assembly).
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --npu-insts-name=%t/pdi_insts.bin --xclbin-name=%t/pdi.xclbin --tmpdir=%t/pdi.prj --cut='{0}.pdi' --checkpoint=%t/pdi.ckpt %s
// RUN: rm -f %t/pdi_insts.bin
// RUN: aiecc --resume=%t/pdi.ckpt/manifest.json
// RUN: cmp %t/ref_insts.bin %t/pdi_insts.bin

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
