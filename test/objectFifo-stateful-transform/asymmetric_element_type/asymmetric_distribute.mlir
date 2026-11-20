//===- asymmetric_distribute.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// A memtile receives @src's 4096-element objects as 2048-element ones (its
// consumer type) and distributes each into two 1024-element outputs: the
// link's buffer is the received object, and each output moves its half.

// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "src__cons__buf0"} : memref<2048xi32>
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_bd(%[[BUF]] : memref<2048xi32> offset = 0 len = 1024)
// CHECK: aie.dma_bd(%[[BUF]] : memref<2048xi32> offset = 1024 len = 1024)
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.dma_bd(%[[BUF]] : memref<2048xi32> offset = 0 len = 1024)
// CHECK: aie.dma_start(MM2S, 1
// CHECK: aie.dma_bd(%[[BUF]] : memref<2048xi32> offset = 1024 len = 1024)

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(0, 3)
    aie.objectfifo @src(%shim, {%mem}, 1 : i32) : !aie.objectfifo<memref<4096xi32>> -> !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @p0(%mem, {%c0}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @p1(%mem, {%c1}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@src] -> [@p0, @p1]([] [0, 1024])
    %core0 = aie.core(%c0) {
      %e = aie.objectfifo.acquire @p0(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @p0(Consume, 1)
      aie.end
    }
    %core1 = aie.core(%c1) {
      %e = aie.objectfifo.acquire @p1(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @p1(Consume, 1)
      aie.end
    }
  }
}
