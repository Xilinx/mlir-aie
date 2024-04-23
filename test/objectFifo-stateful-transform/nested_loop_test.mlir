//===- nested_loop_test.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: April 3rd 2024
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL: aie.device(npu)
//       CHECK:   scf.for
//       CHECK:   {
//       CHECK:     aie.use_lock
//       CHECK:     memref.reinterpret_cast
//       CHECK:     aie.use_lock
//       CHECK:     memref.reinterpret_cast
//       CHECK:     scf.for
//       CHECK:     {
//       CHECK:       scf.for
//       CHECK:       {
//       CHECK:         scf.for
//       CHECK:         {
//       CHECK:           scf.for
//       CHECK:           {
//       CHECK:             scf.for
//       CHECK:             {
//       CHECK:               scf.for
//       CHECK:               {
//       CHECK:                 memref.load
//       CHECK:                 memref.load
//       CHECK:                 memref.load
//       CHECK:                 arith.muli
//       CHECK:                 arith.addi
//       CHECK:                 memref.store
//       CHECK:               }
//       CHECK:             }
//       CHECK:           }
//       CHECK:         }
//       CHECK:       }
//       CHECK:     }
//       CHECK:     aie.use_lock
//       CHECK:     aie.use_lock
//       CHECK:     aie.use_lock
//       CHECK:     memref.reinterpret_cast
//       CHECK:     aie.use_lock
//       CHECK:     memref.reinterpret_cast
//       CHECK:     scf.for
//       CHECK:     {
//       CHECK:       scf.for
//       CHECK:       {
//       CHECK:         scf.for
//       CHECK:         {
//       CHECK:           scf.for
//       CHECK:           {
//       CHECK:             scf.for
//       CHECK:             {
//       CHECK:               scf.for
//       CHECK:               {
//       CHECK:                 memref.load
//       CHECK:                 memref.load
//       CHECK:                 memref.load
//       CHECK:                 arith.muli
//       CHECK:                 arith.addi
//       CHECK:                 memref.store
//       CHECK:               }
//       CHECK:             }
//       CHECK:           }
//       CHECK:         }
//       CHECK:       }
//       CHECK:     }
//       CHECK:     aie.use_lock
//       CHECK:     aie.use_lock
//       CHECK:   }

aie.device(npu) {
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_2 = aie.tile(1, 2)
  %tile_0_2 = aie.tile(0, 2)
  aie.objectfifo @in2(%tile_0_1, {%tile_0_2, %tile_1_2}, 4 : i32) : !aie.objectfifo<memref<32x64xi32, 1>>
  aie.objectfifo @in7(%tile_0_1, {%tile_1_2}, 4 : i32) : !aie.objectfifo<memref<64x32xi32, 1>>
  aie.objectfifo @in8(%tile_1_2, {%tile_0_1}, 4 : i32) : !aie.objectfifo<memref<32x32xi32, 1>>
  %core_1_2 = aie.core(%tile_1_2) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c960 = arith.constant 960 : index
    %0 = aie.objectfifo.acquire @in8(Produce, 1) : !aie.objectfifosubview<memref<32x32xi32, 1>>
    %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xi32, 1>> -> memref<32x32xi32, 1>
    %reinterpret_cast = memref.reinterpret_cast %1 to offset: [0], sizes: [4, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x32xi32, 1> to memref<4x8x4x8xi32, 1>
    aie.objectfifo.release @in2(Consume, 1)
    aie.objectfifo.release @in7(Consume, 1)
    scf.for %arg0 = %c64 to %c960 step %c64 {
      %10 = aie.objectfifo.acquire @in2(Consume, 1) : !aie.objectfifosubview<memref<32x64xi32, 1>>
      %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xi32, 1>> -> memref<32x64xi32, 1>
      %reinterpret_cast_4 = memref.reinterpret_cast %11 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x64xi32, 1> to memref<8x8x4x8xi32, 1>
      %12 = aie.objectfifo.acquire @in7(Consume, 1) : !aie.objectfifosubview<memref<64x32xi32, 1>>
      %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x32xi32, 1>> -> memref<64x32xi32, 1>
      %reinterpret_cast_5 = memref.reinterpret_cast %13 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<64x32xi32, 1> to memref<4x8x8x8xi32, 1>
      scf.for %arg1 = %c0 to %c8 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c4 step %c1 {
              scf.for %arg5 = %c0 to %c8 step %c1 {
                scf.for %arg6 = %c0 to %c8 step %c1 {
                  %14 = memref.load %reinterpret_cast_4[%arg3, %arg1, %arg4, %arg6] : memref<8x8x4x8xi32, 1>
                  %15 = memref.load %reinterpret_cast_5[%arg2, %arg3, %arg6, %arg5] : memref<4x8x8x8xi32, 1>
                  %16 = memref.load %reinterpret_cast[%arg2, %arg1, %arg4, %arg5] : memref<4x8x4x8xi32, 1>
                  %17 = arith.muli %14, %15 : i32
                  %18 = arith.addi %16, %17 : i32
                  memref.store %18, %reinterpret_cast[%arg2, %arg1, %arg4, %arg5] : memref<4x8x4x8xi32, 1>
                }
              }
            }
          }
        }
      }
      aie.objectfifo.release @in2(Consume, 1)
      aie.objectfifo.release @in7(Consume, 1)
    }
    aie.end
  }
}