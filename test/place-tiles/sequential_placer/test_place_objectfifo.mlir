//===- test_place_objectfifo.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// Multiple logical MemTiles merge to same physical tile
// CHECK-LABEL: @multi_fifo_same_tile
module @multi_fifo_same_tile {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // Both mems merge to (0, 1)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)

    // CHECK: aie.objectfifo @of1(%[[MEM]], {%[[CORE]]}, 2 : i32)
    aie.objectfifo @of1 (%mem1, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK: aie.objectfifo @of2(%[[MEM]], {%[[CORE]]}, 2 : i32)
    aie.objectfifo @of2 (%mem2, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK-NOT: aie.tile(1, 1)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Multiple logical ShimNOCTiles merge when feeding same core
// CHECK-LABEL: @worker_multiple_inputs
module @worker_multiple_inputs {
  aie.device(npu1) {
    // Both shims should merge to (0, 0)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @in1 (%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim2, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
    // CHECK-NOT: aie.tile(1, 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// ObjectFifo with multiple consumers (broadcast)
// CHECK-LABEL: @multi_consumer
module @multi_consumer {
  aie.device(npu1) {
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    // CHECK-DAG: %[[CORE1:.*]] = aie.tile(0, 2)
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[CORE2:.*]] = aie.tile(0, 3)
    %core2 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @broadcast (%shim, {%core1, %core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Linked fifos: link point memtile is the shared tile
// CHECK-LABEL: @objectfifo_link
module @objectfifo_link {
  aie.device(npu1) {
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    // CHECK-DAG: %[[CORE0:.*]] = aie.tile(0, 2)
    // CHECK-DAG: %[[CORE1:.*]] = aie.tile(0, 3)

    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core0 = aie.logical_tile<CoreTile>(?, ?)
    %core1 = aie.logical_tile<CoreTile>(?, ?)

    // shim -> mem (splits to 2 outputs)
    aie.objectfifo @inA (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memA0 (%mem, {%core0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memA1 (%mem, {%core1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>

    // Link: 1 input, 2 outputs
    aie.objectfifo.link [@inA] -> [@memA0, @memA1] ([] [0, 1024])
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Exactly fills 2 input channels on core tile (max capacity)
// CHECK-LABEL: @exactly_fills_2_input_channels
module @exactly_fills_2_input_channels {
  aie.device(npu1) {
    // Both shims should merge to (0, 0)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)

    // 2 input fifos = exactly 2 DMA input channels (max for core tile)
    aie.objectfifo @in1 (%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim2, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
    // CHECK-NOT: aie.tile(1, 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Channel overflow forces third shim placement to next column
// CHECK-LABEL: @channel_overflow_next_column
module @channel_overflow_next_column {
  aie.device(npu1) {
    // First two shims merge to (0, 0), using 2 output channels
    // CHECK-DAG: %[[SHIM1:.*]] = aie.tile(0, 0)
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    // Third shim must go to column 1 due to channel exhaustion
    // CHECK-DAG: %[[SHIM2:.*]] = aie.tile(1, 0)
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)

    %core1 = aie.logical_tile<CoreTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)

    // shim1 and shim2 merge, using 2 output channels total
    aie.objectfifo @of1 (%shim1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%shim2, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // shim3 overflows to next column
    aie.objectfifo @of3 (%shim3, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    aie.core(%core3) { aie.end }
    // CHECK-NOT: aie.tile(2, 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Exactly fills 2 output channels on core tile (max capacity)
// CHECK-LABEL: @exactly_fills_2_output_channels
module @exactly_fills_2_output_channels {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // Both mems should merge to (0, 1)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)

    // 2 output fifos = exactly 2 DMA output channels (max for core tile)
    aie.objectfifo @out1 (%core, {%mem1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2 (%core, {%mem2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
    // CHECK-NOT: aie.tile(1, 1)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Exactly fills both input and output channels on core tile
// CHECK-LABEL: @exactly_fills_all_core_channels
module @exactly_fills_all_core_channels {
  aie.device(npu1) {
    // Both shims should merge to (0, 0)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // Both mems should merge to (0, 1)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)

    // 2 inputs + 2 outputs = all 4 DMA channels used
    aie.objectfifo @in1 (%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim2, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1 (%core, {%mem1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2 (%core, {%mem2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
    // CHECK-NOT: aie.tile(1, 0)
    // CHECK-NOT: aie.tile(1, 1)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Multiple logical ShimNOCTiles merge into one physical tile (exactly fills capacity)
// CHECK-LABEL: @exactly_fills_shim_channels
module @exactly_fills_shim_channels {
  aie.device(npu1) {
    // All 4 logical shims should merge into one physical shim (0, 0)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    // CHECK-NOT: aie.tile(1, 0)
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim4 = aie.logical_tile<ShimNOCTile>(?, ?)

    %core1 = aie.logical_tile<CoreTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)
    %core4 = aie.logical_tile<CoreTile>(?, ?)

    // 2 outputs (shim -> cores) + 2 inputs (cores -> shim) = exactly 4 channels
    aie.objectfifo @toCore1 (%shim1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @toCore2 (%shim2, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @fromCore3 (%core3, {%shim3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @fromCore4 (%core4, {%shim4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    aie.core(%core3) { aie.end }
    aie.core(%core4) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Multiple logical MemTiles merge into one physical tile (exactly fills 6 output channels)
// CHECK-LABEL: @exactly_fills_memtile_channels
module @exactly_fills_memtile_channels {
  aie.device(npu1) {
    // All 6 logical memtiles should merge into one physical memtile (0, 1)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    // CHECK-NOT: aie.tile(1, 1)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)
    %mem4 = aie.logical_tile<MemTile>(?, ?)
    %mem5 = aie.logical_tile<MemTile>(?, ?)
    %mem6 = aie.logical_tile<MemTile>(?, ?)

    %core1 = aie.logical_tile<CoreTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)
    %core4 = aie.logical_tile<CoreTile>(?, ?)

    // 6 outputs from memtiles to cores = exactly 6 output channels (MemTile max)
    aie.objectfifo @of1 (%mem1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%mem2, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of3 (%mem3, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of4 (%mem4, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of5 (%mem5, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of6 (%mem6, {%core4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    aie.core(%core3) { aie.end }
    aie.core(%core4) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Partial capacity remaining but not enough for next request
// CHECK-LABEL: @memtile_partial_overflow
module @memtile_partial_overflow {
  aie.device(npu1) {
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim4 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim5 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim6 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim7 = aie.logical_tile<ShimNOCTile>(?, ?)

    // First 5 memtiles merge to (0, 1), using 5 input channels
    // CHECK-DAG: %[[MEM1:.*]] = aie.tile(0, 1)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)
    %mem4 = aie.logical_tile<MemTile>(?, ?)
    %mem5 = aie.logical_tile<MemTile>(?, ?)
    // Next 2 memtiles need 2 input channels, but only 1 remains on (0, 1)
    // So they go to (1, 1)
    // CHECK-DAG: %[[MEM2:.*]] = aie.tile(1, 1)
    %mem6 = aie.logical_tile<MemTile>(?, ?)
    %mem7 = aie.logical_tile<MemTile>(?, ?)

    %core = aie.logical_tile<CoreTile>(?, ?)

    // 5 inputs to first memtile group
    aie.objectfifo @in1 (%shim1, {%mem1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim2, {%mem2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in3 (%shim3, {%mem3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in4 (%shim4, {%mem4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in5 (%shim5, {%mem5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // 2 inputs to second memtile group (overflow)
    aie.objectfifo @in6 (%shim6, {%mem6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in7 (%shim7, {%mem7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // CHECK-NOT: aie.tile(2, 1)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Linked fifos: memtile placed at averaged column of connected cores
// CHECK-LABEL: @linked_fifos_averaged_column
module @linked_fifos_averaged_column {
  aie.device(npu1) {
    // Cores in columns 0 and 2, MemTile should be at column 1 (average)
    // CHECK-DAG: %[[CORE0:.*]] = aie.tile(0, 2)
    %core0 = aie.logical_tile<CoreTile>(0, 2)
    // CHECK-DAG: %[[CORE1:.*]] = aie.tile(2, 2)
    %core1 = aie.logical_tile<CoreTile>(2, 2)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(1, 1)
    // Link point: mem is shared tile for all linked fifos
    %mem = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(1, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)

    aie.objectfifo @inA (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @toCore0 (%mem, {%core0}, 2 : i32) : !aie.objectfifo<memref<512xbf16>>
    aie.objectfifo @toCore1 (%mem, {%core1}, 2 : i32) : !aie.objectfifo<memref<512xbf16>>

    aie.objectfifo.link [@inA] -> [@toCore0, @toCore1] ([] [0, 512])

    aie.core(%core0) { aie.end }
    aie.core(%core1) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}
