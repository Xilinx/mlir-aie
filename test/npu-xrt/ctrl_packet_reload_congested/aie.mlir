//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// @cg is loaded by control packets, so it must route the column control
// overlay the same way the standalone overlay does (#3837). Its circuit flows
// crowd the memtile's north ports. If that moves the overlay's routes, loading
// @cg reprograms the switches its own control packets travel through, and the
// run never finishes. Cores (0,2) to (0,5) add 2 to 5 to their quarter of the
// input.
module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg : memref<16xi32>) {
      aiex.configure @cg {
        aiex.run @cg_sequence (%arg) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu2) @cg {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t10 = aie.tile(1, 0)
    %t11 = aie.tile(1, 1)
    %t12 = aie.tile(1, 2)
    %k11 = aie.buffer(%t11) {sym_name = "k11"} : memref<4xi32>
    %k12 = aie.buffer(%t12) {sym_name = "k12"} : memref<4xi32>
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.objectfifo @in(%t10, {%t01}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%t01, {%t10}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @i2(%t01, {%t02}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @o2(%t02, {%t01}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @i3(%t01, {%t03}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @o3(%t03, {%t01}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @i4(%t01, {%t04}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @o4(%t04, {%t01}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @i5(%t01, {%t05}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @o5(%t05, {%t01}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo.link [@in] -> [@i2, @i3, @i4, @i5] ([] [0, 4, 8, 12])
    aie.objectfifo.link [@o2, @o3, @o4, @o5] -> [@out] ([0, 4, 8, 12] [])
    aie.flow(%t01, DMA : 5, %t02, DMA : 1)
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %k = arith.constant 2 : i32
      %bi = aie.objectfifo.acquire @i2(Consume, 1) : memref<4xi32>
      %bo = aie.objectfifo.acquire @o2(Produce, 1) : memref<4xi32>
      scf.for %i = %c0 to %cn step %c1 {
        %x = memref.load %bi[%i] : memref<4xi32>
        %y = arith.addi %x, %k : i32
        memref.store %y, %bo[%i] : memref<4xi32>
      }
      aie.objectfifo.release @i2(Consume, 1)
      aie.objectfifo.release @o2(Produce, 1)
      aie.end
    }
    aie.core(%t03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %k = arith.constant 3 : i32
      %bi = aie.objectfifo.acquire @i3(Consume, 1) : memref<4xi32>
      %bo = aie.objectfifo.acquire @o3(Produce, 1) : memref<4xi32>
      scf.for %i = %c0 to %cn step %c1 {
        %x = memref.load %bi[%i] : memref<4xi32>
        %y = arith.addi %x, %k : i32
        memref.store %y, %bo[%i] : memref<4xi32>
      }
      aie.objectfifo.release @i3(Consume, 1)
      aie.objectfifo.release @o3(Produce, 1)
      aie.end
    }
    aie.core(%t04) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %k = arith.constant 4 : i32
      %bi = aie.objectfifo.acquire @i4(Consume, 1) : memref<4xi32>
      %bo = aie.objectfifo.acquire @o4(Produce, 1) : memref<4xi32>
      scf.for %i = %c0 to %cn step %c1 {
        %x = memref.load %bi[%i] : memref<4xi32>
        %y = arith.addi %x, %k : i32
        memref.store %y, %bo[%i] : memref<4xi32>
      }
      aie.objectfifo.release @i4(Consume, 1)
      aie.objectfifo.release @o4(Produce, 1)
      aie.end
    }
    aie.core(%t05) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %k = arith.constant 5 : i32
      %bi = aie.objectfifo.acquire @i5(Consume, 1) : memref<4xi32>
      %bo = aie.objectfifo.acquire @o5(Produce, 1) : memref<4xi32>
      scf.for %i = %c0 to %cn step %c1 {
        %x = memref.load %bi[%i] : memref<4xi32>
        %y = arith.addi %x, %k : i32
        memref.store %y, %bo[%i] : memref<4xi32>
      }
      aie.objectfifo.release @i5(Consume, 1)
      aie.objectfifo.release @o5(Produce, 1)
      aie.end
    }
    aie.runtime_sequence @cg_sequence(%a : memref<16xi32>) {
      %ti = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %to = aiex.dma_configure_task_for @out {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti)
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
    }
  }
}
