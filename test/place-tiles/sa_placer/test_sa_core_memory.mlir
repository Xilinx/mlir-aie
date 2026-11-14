//===- test_sa_core_memory.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-assign-buffer-addresses))' %t -o /dev/null

// The SA sizes a core's memory as aie-assign-buffer-addresses does, so what
// does not fit goes to a neighbor (objectfifo.allocate) before buffers are
// placed.

// An i4 object of 20000 elements takes 20000 bytes, one per element, as the
// pool and buffer sizes count it, not 10000: the two intra-tile fifos need
// 80000 bytes. Of the two equal fifos, the one the core acquires first
// moves, whichever is declared first.
// CHECK-LABEL: @subbyte_core
// CHECK:       aie.objectfifo.allocate @a(
// CHECK-NOT:   aie.objectfifo.allocate
// CHECK-LABEL: @subbyte_core_declared_backwards
// CHECK:       aie.objectfifo.allocate @a(
// CHECK-NOT:   aie.objectfifo.allocate

// A core's measured data section (10000 bytes) is reserved as measured, not
// as the SA's 3 KiB default.
// CHECK-LABEL: @measured_data
// CHECK:       aie.objectfifo.allocate

module @subbyte_core {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%core, {%core}, 2 : i32) : !aie.objectfifo<memref<20000xi4>>
    aie.objectfifo @b(%core, {%core}, 2 : i32) : !aie.objectfifo<memref<20000xi4>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<20000xi4>
      aie.objectfifo.release @a(Produce, 1)
      %1 = aie.objectfifo.acquire @a(Consume, 1) : memref<20000xi4>
      aie.objectfifo.release @a(Consume, 1)
      %2 = aie.objectfifo.acquire @b(Produce, 1) : memref<20000xi4>
      aie.objectfifo.release @b(Produce, 1)
      %3 = aie.objectfifo.acquire @b(Consume, 1) : memref<20000xi4>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

// -----

module @subbyte_core_declared_backwards {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @b(%core, {%core}, 2 : i32) : !aie.objectfifo<memref<20000xi4>>
    aie.objectfifo @a(%core, {%core}, 2 : i32) : !aie.objectfifo<memref<20000xi4>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<20000xi4>
      aie.objectfifo.release @a(Produce, 1)
      %1 = aie.objectfifo.acquire @a(Consume, 1) : memref<20000xi4>
      aie.objectfifo.release @a(Consume, 1)
      %2 = aie.objectfifo.acquire @b(Produce, 1) : memref<20000xi4>
      aie.objectfifo.release @b(Produce, 1)
      %3 = aie.objectfifo.acquire @b(Consume, 1) : memref<20000xi4>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

// -----

module @measured_data {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%core, {%core}, 2 : i32) : !aie.objectfifo<memref<14000xi8>>
    aie.objectfifo @b(%core, {%core}, 2 : i32) : !aie.objectfifo<memref<14000xi8>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<14000xi8>
      aie.objectfifo.release @a(Produce, 1)
      %1 = aie.objectfifo.acquire @a(Consume, 1) : memref<14000xi8>
      aie.objectfifo.release @a(Consume, 1)
      %2 = aie.objectfifo.acquire @b(Produce, 1) : memref<14000xi8>
      aie.objectfifo.release @b(Produce, 1)
      %3 = aie.objectfifo.acquire @b(Consume, 1) : memref<14000xi8>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    } {measured_data_size = 10000 : i32}
  }
}
