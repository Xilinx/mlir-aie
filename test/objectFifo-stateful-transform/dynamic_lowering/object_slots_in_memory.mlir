//===- object_slots_in_memory.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's objects stay in their slots wherever they are still picked at run
// time, rather than being carried through control flow as memrefs.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll -split-input-file %s | FileCheck %s

// A static core whose inner loop runs 3 times over a depth-2 objectFifo: every
// outer iteration leaves the objects rotated by one, so they do not fold.

// CHECK-LABEL: aie.device(npu2)
// CHECK:     %[[CORE:.*]] = aie.core
// CHECK:       %[[S0:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       memref.store %{{.*}}buff_0, %[[S0]][]
// CHECK:       %[[S1:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       memref.store %{{.*}}buff_1, %[[S1]][]
// CHECK-NOT:   iter_args
// CHECK:       scf.for
// CHECK:         %[[OBJ:.*]] = memref.load %[[S0]][] : memref<memref<8xi8>>
// CHECK:         memref.load %[[OBJ]]
// CHECK:       aie.end
module {
  aie.device(npu2) {
    %t1 = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    %buf = aie.buffer(%t2) {sym_name = "buf"} : memref<8xi8>
    aie.objectfifo @fifo(%t1, {%t2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>
    %core = aie.core(%t2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %cmax = arith.constant 9223372036854775807 : index
      scf.for %o = %c0 to %cmax step %c1 {
        scf.for %i = %c0 to %c3 step %c1 {
          %x = aie.objectfifo.acquire @fifo(Consume, 1) : memref<8xi8>
          %v = memref.load %x[%c0] : memref<8xi8>
          memref.store %v, %buf[%c0] : memref<8xi8>
          aie.objectfifo.release @fifo(Consume, 1)
        }
      }
      aie.end
    } {dynamic_objfifo_lowering = false}
  }
}

// -----

// A dynamic core whose loop is a CFG cycle keeps its objects in their slots
// rather than threading them through block arguments.

// CHECK-LABEL: aie.device(npu2)
// CHECK:     aie.core
// CHECK:       %[[S0:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       %[[S1:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       cf.br ^[[LOOP:.*]](%{{.*}} : index)
// CHECK:     ^[[LOOP]](%{{.*}}: index):
// CHECK:       %[[OBJ:.*]] = memref.load %[[S0]][] : memref<memref<8xi8>>
// CHECK:       memref.load %[[OBJ]]
module {
  aie.device(npu2) {
    %t1 = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    %buf = aie.buffer(%t2) {sym_name = "buf"} : memref<8xi8>
    aie.objectfifo @fifo(%t1, {%t2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>
    %core = aie.core(%t2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      cf.br ^loop(%c0 : index)
    ^loop(%i: index):
      %o = aie.objectfifo.acquire @fifo(Consume, 1) : memref<8xi8>
      %v = memref.load %o[%c0] : memref<8xi8>
      memref.store %v, %buf[%c0] : memref<8xi8>
      aie.objectfifo.release @fifo(Consume, 1)
      %n = arith.addi %i, %c1 : index
      %cond = arith.cmpi slt, %n, %c14 : index
      cf.cond_br %cond, ^loop(%n : index), ^exit
    ^exit:
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
