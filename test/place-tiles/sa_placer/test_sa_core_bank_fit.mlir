//===- test_sa_core_bank_fit.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-assign-buffer-addresses))' %t -o /dev/null

// The SA fits a core's buffers by aie-assign-buffer-addresses' own search.
// Four buffers pinned one per bank leave 58 KB of room by bytes, but no
// run of 14000 bytes, so the fifo's object goes to a neighbour.
// CHECK: aie.objectfifo.allocate @a

module @bank_fragments {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    %b0 = aie.buffer(%core) {mem_bank = 0 : i32, sym_name = "b0"} : memref<10000xi8>
    %b1 = aie.buffer(%core) {mem_bank = 1 : i32, sym_name = "b1"} : memref<10000xi8>
    %b2 = aie.buffer(%core) {mem_bank = 2 : i32, sym_name = "b2"} : memref<10000xi8>
    %b3 = aie.buffer(%core) {mem_bank = 3 : i32, sym_name = "b3"} : memref<10000xi8>
    aie.objectfifo @a(%core, {%core}, 1 : i32) : !aie.objectfifo<memref<14000xi8>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<14000xi8>
      aie.objectfifo.release @a(Produce, 1)
      %1 = aie.objectfifo.acquire @a(Consume, 1) : memref<14000xi8>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
}
