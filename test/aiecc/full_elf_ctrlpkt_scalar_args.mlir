//===- full_elf_ctrlpkt_scalar_args.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Only a runtime_sequence's memref arguments are host buffers, so a full ELF's
// kernel arguments and its control buffer's slot count memrefs, not scalars.
// Here %a, %n : i32 and %b take host buffers 0 and 1, and ctrl-packet-to-DMA
// appends the control buffer as host buffer 2: the slot its DMAs patch, the
// patch info's xrt_id, and the last of the full ELF's three kernel arguments.

// REQUIRES: peano

// RUN: rm -rf %t.prj
// RUN: aiecc --get-full-elf --load-pdi-to-ctrl-pkt --dump-intermediates --tmpdir=%t.prj %s
// RUN: FileCheck %s --check-prefix=SEQ < %t.prj/npu_seq_main_sequence.mlir
// RUN: FileCheck %s --check-prefix=PATCH < %t.prj/full_elf_main_sequence.patch_info.json
// RUN: FileCheck %s --check-prefix=ARGS < %t.prj/full_elf_config.json

// SEQ: aie.runtime_sequence(%arg0: memref<64xi32>{{.*}}, %arg1: i32{{.*}}, %arg2: memref<64xi32>{{.*}}, %arg3: memref<?xi32>
// SEQ: aiex.npu.address_patch{{.*}}arg_idx = 2 : i32
// SEQ: aiex.npu.address_patch{{.*}}arg_idx = 1 : i32

// PATCH: "xrt_id": 2

// ARGS: "name": "arg_2"
// ARGS-NOT: "name": "arg_3"

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%a : memref<64xi32>, %n : i32, %b : memref<64xi32>) {
      aiex.npu.load_pdi {device_ref = @cfg}
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @out, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @out(%t00, S2MM, 0)
  }
  aie.device(npu2) @cfg {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.flow(%t01, DMA : 0, %t02, DMA : 0)
    aie.flow(%t02, DMA : 0, %t01, DMA : 0)
  }
}
