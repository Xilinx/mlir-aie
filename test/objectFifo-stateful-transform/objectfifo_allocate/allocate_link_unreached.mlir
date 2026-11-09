//===- allocate_link_unreached.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s

// A link's input that does not own the link's pool, whose allocate's delegate
// its producer cannot reach, moves by DMA with its own objects, so the pool's
// larger object type does not concern it.

// CHECK-LABEL: module @larger_output_far
// CHECK:     %[[A:.*]] = aie.tile(0, 2)
// CHECK-DAG: aie.buffer(%[[A]]) {{.*}}sym_name = "in0__buf0"} : memref<1024xi32>
// CHECK-DAG: aie.buffer(%[[A]]) {{.*}}sym_name = "in0__buf1"} : memref<1024xi32>
// CHECK:     %[[L:.*]] = aie.tile(0, 4)
// CHECK-DAG: aie.buffer(%[[L]]) {{.*}}sym_name = "out__buf3"} : memref<2048xi32>
// CHECK-DAG: aie.flow(%[[A]], DMA : 0, %[[L]], DMA : 0)
module @larger_output_far {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 4)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 4 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @join_far
// CHECK:     %[[A:.*]] = aie.tile(0, 2)
// CHECK-DAG: aie.buffer(%[[A]]) {{.*}}sym_name = "in0__buf0"} : memref<1024xi32>
// CHECK:     %[[L:.*]] = aie.tile(0, 5)
// CHECK-DAG: aie.buffer(%[[L]]) {{.*}}sym_name = "out__buf1"} : memref<2048xi32>
// CHECK-DAG: aie.flow(%[[A]], DMA : 0, %[[L]], DMA : 0)
module @join_far {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %l = aie.tile(0, 5)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @in1(%b, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 1024] [])
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @in1(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in1(Produce, 1)
      aie.end
    }
  }
}
