//===- bd_halves_walk_whole.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-bd-ids %s -o /dev/null

// Six links through one mem tile, pools 1, 1, 1, 1, 1 and 11 deep: 32 BD ids
// in all, which would fit the tile's two halves of 24, but channels go to
// ends first-free, and the deep pool's ends take channel 5 in each
// direction, both in the odd half. Short of BD ids there, the allocator
// walks that pool whole, one BD a side, and BD ids assign.
// CHECK: aie.objectfifo.pool @in5__cons__pool(%{{.*}}) {buffers = [@in5__cons__buf0], contiguous, depth = 11 : i32

module @halves {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %m = aie.tile(0, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %t2 = aie.tile(2, 2)
    %t3 = aie.tile(3, 2)
    %t4 = aie.tile(4, 2)
    %t5 = aie.tile(5, 2)
    aie.objectfifo @in0(%s0, {%m}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out0(%m, {%t0}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in0] -> [@out0] ([] [])
    aie.objectfifo @in1(%s0, {%m}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out1(%m, {%t1}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in1] -> [@out1] ([] [])
    aie.objectfifo @in2(%s1, {%m}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out2(%m, {%t2}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in2] -> [@out2] ([] [])
    aie.objectfifo @in3(%s1, {%m}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out3(%m, {%t3}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in3] -> [@out3] ([] [])
    aie.objectfifo @in4(%s2, {%m}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out4(%m, {%t4}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in4] -> [@out4] ([] [])
    aie.objectfifo @in5(%s2, {%m}, [1 : i32, 11 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out5(%m, {%t5}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in5] -> [@out5] ([] [])
    aie.core(%t0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %k = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @out0(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @out0(Consume, 1)
      }
      aie.end
    }
    aie.core(%t1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %k = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @out1(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @out1(Consume, 1)
      }
      aie.end
    }
    aie.core(%t2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %k = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @out2(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @out2(Consume, 1)
      }
      aie.end
    }
    aie.core(%t3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %k = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @out3(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @out3(Consume, 1)
      }
      aie.end
    }
    aie.core(%t4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %k = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @out4(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @out4(Consume, 1)
      }
      aie.end
    }
    aie.core(%t5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %k = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @out5(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @out5(Consume, 1)
      }
      aie.end
    }
  }
}
