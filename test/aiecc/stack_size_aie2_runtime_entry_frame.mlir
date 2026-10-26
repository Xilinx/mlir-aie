//===- stack_size_aie2_runtime_entry_frame.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The npu1 counterpart of stack_size_counts_runtime_entry_frame.mlir, which
// covers npu2. Peano must compile aie2's crt1.o with -fstack-size-section for
// the linked core to report the frame of `_main_init`.
//
// This core calls no kernel and its body needs no frame, so 32 is the runtime
// entry frame alone. The requirement drops to 0 when the frame goes
// unreported, and every npu1 core then undercounts by 32 bytes.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: cd %t.d && %aiecc --get=measured_stack_sizes.mlir --output-dir=%t.out %s
// RUN: FileCheck %s --input-file %t.out/measured_stack_sizes.mlir

// CHECK: measured_stack_size = 32

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<64xi8>
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 1024 : i32 }

    aie.runtime_sequence(%out : memref<64xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
