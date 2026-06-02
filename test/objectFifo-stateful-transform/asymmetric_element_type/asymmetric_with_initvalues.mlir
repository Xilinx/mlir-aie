//===- asymmetric_with_initvalues.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// Asymmetric ObjectFifo with initValues and depth=2.
// Mobilenet weight loading pattern: 2 large buffers with static weight
// data on MemTile, consumer receives small chunks on CoreTile.

// Consumer buffers use the small type
// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "wts_cons_buff_0"} : memref<10xi32>
// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "wts_cons_buff_1"} : memref<10xi32>

// Producer buffers have initial values
// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "wts_buff_0"} : memref<40xi32> = dense<1>
// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "wts_buff_1"} : memref<40xi32> = dense<2>

// Producer lock: init = 0 (depth - initValues count = 0)
// CHECK-DAG: aie.lock(%{{.*}}) {init = 0 : i32, sym_name = "wts_prod_lock_0"}
// Consumer lock: init = 2 (initValues count)
// CHECK-DAG: aie.lock(%{{.*}}) {init = 2 : i32, sym_name = "wts_cons_lock_0"}

// MM2S: 2 BDs for depth=2, each 40 elements
// CHECK: aie.memtile_dma
// CHECK: aie.dma_bd(%{{.*}} : memref<40xi32>, 0, 40)
// CHECK: aie.dma_bd(%{{.*}} : memref<40xi32>, 0, 40)

// S2MM: 2 BDs (consumer depth=2), 10 elements each
// CHECK: aie.mem
// CHECK: aie.dma_bd(%{{.*}} : memref<10xi32>, 0, 10)
// CHECK: aie.dma_bd(%{{.*}} : memref<10xi32>, 0, 10)

module {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %ct = aie.tile(0, 2)

    aie.objectfifo @wts(%mt, {%ct}, 2 : i32)
      : !aie.objectfifo<memref<40xi32>>
      -> !aie.objectfifo<memref<10xi32>>
      = [dense<1> : memref<40xi32>, dense<2> : memref<40xi32>]

    %c = aie.core(%ct) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %sv = aie.objectfifo.acquire @wts(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %elem = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        aie.objectfifo.release @wts(Consume, 1)
      }
      aie.end
    }
  }
}
