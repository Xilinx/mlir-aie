//===- test_sa_memory.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify SA placer considers memory capacity during placement.
// CoreTile has 64KB local memory (NPU2). MemTile has 512KB.

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// Small buffers: should place without issues
// CHECK-LABEL: @small_buffers
module @small_buffers {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)

    // 2 x 256 x i32 = 2KB total -- well within 64KB
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%core) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Multiple ObjectFifos on one core, all within capacity
// CHECK-LABEL: @multi_fifo_within_capacity
module @multi_fifo_within_capacity {
  aie.device(npu2) {
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)

    // 4 fifos x 2 depth x 1024 x i32 = 32KB -- within 64KB
    aie.objectfifo @in1(%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @in2(%shim2, {%core}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out1(%core, {%shim1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out2(%core, {%shim2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>

    aie.core(%core) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Pipeline of cores: memory spread across tiles
// CHECK-LABEL: @pipeline_spread
module @pipeline_spread {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)

    // 2 x 4096 x i32 = 32KB per fifo, but spread across 3 tiles
    aie.objectfifo @in(%shim, {%c0}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @p01(%c0, {%c1}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @p12(%c1, {%c2}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @out(%c2, {%shim}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>

    aie.core(%c0) { aie.end }
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Different element types: bf16 (2 bytes), i8 (1 byte)
// CHECK-LABEL: @mixed_element_types
module @mixed_element_types {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)

    // bf16: 2 x 1024 x 2bytes = 4KB
    aie.objectfifo @bf16_fifo(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    // i8: 2 x 2048 x 1byte = 4KB
    aie.objectfifo @i8_fifo(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

    aie.core(%core) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// MemTile with larger buffers (512KB capacity)
// CHECK-LABEL: @memtile_large_buffers
module @memtile_large_buffers {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)

    // Large L3->L2 buffer through MemTile: 2 x 32768 x i32 = 256KB
    aie.objectfifo @inL3(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<32768xi32>>
    aie.objectfifo @inL2_0(%mem, {%c0}, 2 : i32) : !aie.objectfifo<memref<16384xi32>>
    aie.objectfifo @inL2_1(%mem, {%c1}, 2 : i32) : !aie.objectfifo<memref<16384xi32>>
    aie.objectfifo.link [@inL3] -> [@inL2_0, @inL2_1]([] [0, 16384])

    aie.core(%c0) { aie.end }
    aie.core(%c1) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}
