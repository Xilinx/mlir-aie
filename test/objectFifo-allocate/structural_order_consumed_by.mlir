// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One design written twice, the second renaming the outputs so their names
// sort the other way (r -> z, s -> a). Each shim input's consumed_by names its
// output, and the name must not steer which channel the input gets.

// CHECK-LABEL: module @declared
// CHECK-DAG: aie.shim_dma_allocation @p_shim_alloc(%{{.*}}, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @r_shim_alloc
// CHECK-DAG: aie.shim_dma_allocation @q_shim_alloc(%{{.*}}, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @s_shim_alloc
// CHECK-DAG: aie.shim_dma_allocation @r_shim_alloc(%{{.*}}, S2MM, 0)
// CHECK-DAG: aie.shim_dma_allocation @s_shim_alloc(%{{.*}}, S2MM, 1)

// CHECK-LABEL: module @renamed
// CHECK-DAG: aie.shim_dma_allocation @p_shim_alloc(%{{.*}}, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @z_shim_alloc
// CHECK-DAG: aie.shim_dma_allocation @q_shim_alloc(%{{.*}}, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @a_shim_alloc
// CHECK-DAG: aie.shim_dma_allocation @z_shim_alloc(%{{.*}}, S2MM, 0)
// CHECK-DAG: aie.shim_dma_allocation @a_shim_alloc(%{{.*}}, S2MM, 1)

module @declared {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %core2 = aie.tile(0, 2)
    %core3 = aie.tile(0, 3)
    aie.objectfifo @p(%shim, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q(%shim, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @r(%core2, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @s(%core3, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a = aie.objectfifo.acquire @p(Consume, 1) : memref<16xi32>
        %b = aie.objectfifo.acquire @r(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @p(Consume, 1)
        aie.objectfifo.release @r(Produce, 1)
      }
      aie.end
    }
    aie.core(%core3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a = aie.objectfifo.acquire @q(Consume, 1) : memref<16xi32>
        %b = aie.objectfifo.acquire @s(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @q(Consume, 1)
        aie.objectfifo.release @s(Produce, 1)
      }
      aie.end
    }
  }
}

// -----

module @renamed {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %core2 = aie.tile(0, 2)
    %core3 = aie.tile(0, 3)
    aie.objectfifo @p(%shim, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q(%shim, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z(%core2, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%core3, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a = aie.objectfifo.acquire @p(Consume, 1) : memref<16xi32>
        %b = aie.objectfifo.acquire @z(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @p(Consume, 1)
        aie.objectfifo.release @z(Produce, 1)
      }
      aie.end
    }
    aie.core(%core3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a = aie.objectfifo.acquire @q(Consume, 1) : memref<16xi32>
        %b = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @q(Consume, 1)
        aie.objectfifo.release @a(Produce, 1)
      }
      aie.end
    }
  }
}
