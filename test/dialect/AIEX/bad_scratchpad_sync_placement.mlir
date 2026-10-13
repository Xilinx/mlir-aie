//===- bad_scratchpad_sync_placement.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The parameter sync goes after the last top-level load_pdi, so a DMA that
// reads a parameter before then would run before the scratchpad is created.
// A sync placed by hand is left where it is.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-lower-scratchpad-parameters %s

aiex.scratchpad_parameter @off : i32
aie.device(npu2) @empty { }
aie.device(npu2) @dev {
  aie.runtime_sequence @offset_before_pdi(%arg0 : memref<4096xi32>) {
    // expected-error @+1 {{'aiex.npu.dma_memcpy_nd' op reads a scratchpad parameter before the last aiex.npu.load_pdi of its runtime sequence}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @dma, offset_parameter = @off} : memref<4096xi32>
    aiex.npu.load_pdi {device_ref = @dev}
  }
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @dma(%t, MM2S, 0)
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) @empty { }
aie.device(npu2) @dev {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @length_between_pdis(%arg0 : memref<4096xi32>) {
    aiex.npu.load_pdi {device_ref = @empty}
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op reads a scratchpad parameter before the last aiex.npu.load_pdi of its runtime sequence}}
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64) {bd_id = 3 : i32, length_parameter = @n, length_unit = 16 : i32}
      aie.end
    }
    aiex.npu.load_pdi {device_ref = @dev}
    aiex.dma_start_task(%task)
  }
}

// -----

aiex.scratchpad_parameter @off : i32
aie.device(npu2) @dev {
  aie.runtime_sequence @placed_by_hand(%arg0 : memref<4096xi32>) {
    aiex.sync_scratchpad_parameters_from_host
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @dma, offset_parameter = @off} : memref<4096xi32>
    aiex.npu.load_pdi {device_ref = @dev}
  }
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @dma(%t, MM2S, 0)
}
