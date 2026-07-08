//===- cpp_multi_device_sequence.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// Test device and sequence filtering (mirrors test/npu-xrt/add_one_two pattern)
// Each device is compiled separately with --device-name, similar to how
// the Python aiecc.py is used in production multi-device flows.

// RUN: aiecc --no-xchesscc --no-xbridge --device-name=device1 --verbose --tmpdir=%t.dev1 %s 2>&1 | FileCheck %s --check-prefix=DEV1
// RUN: aiecc --no-xchesscc --no-xbridge --device-name=device2 --verbose --tmpdir=%t.dev2 %s 2>&1 | FileCheck %s --check-prefix=DEV2
// RUN: aiecc --no-xchesscc --no-xbridge --device-name=device1 --sequence-name=seq_a --aie-generate-npu-insts --verbose --tmpdir=%t.seq_a %s 2>&1 | FileCheck %s --check-prefix=DEV1_SEQ_A
// RUN: aiecc --no-xchesscc --no-xbridge --verbose --tmpdir=%t.all %s 2>&1 | FileCheck %s --check-prefix=ALL

// The kernels.json host-buffer (boN) count must come from the SELECTED
// sequence: seq_a has 6 host args (bo0..bo5), seq_b has 7 (bo0..bo6).
// RUN: aiecc --no-xchesscc --no-xbridge --device-name=device1 --sequence-name=seq_a --aie-generate-xclbin --tmpdir=%t.count_a %s
// RUN: FileCheck %s --check-prefix=COUNT_A --input-file=%t.count_a/device1_kernels.json
// RUN: aiecc --no-xchesscc --no-xbridge --device-name=device1 --sequence-name=seq_b --aie-generate-xclbin --tmpdir=%t.count_b %s
// RUN: FileCheck %s --check-prefix=COUNT_B --input-file=%t.count_b/device1_kernels.json
// RUN: aie-opt -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" %s -o %t.ctrlpkt_overlay.mlir && aiecc --no-xchesscc --no-xbridge --device-name=device1 --aie-generate-ctrlpkt --verbose --tmpdir=%t.ctrlpkt %t.ctrlpkt_overlay.mlir 2>&1 | FileCheck %s --check-prefix=CTRLPKT_DEV1

// DEV1: Removing non-matching device: device2
// DEV1: Processing device: device1
// DEV1: No cores to compile in device device1
// DEV1-NOT: Processing device: device2
// DEV1: Compilation completed successfully

// DEV2: Removing non-matching device: device1
// DEV2: Processing device: device2
// DEV2: No cores to compile in device device2
// DEV2-NOT: Processing device: device1
// DEV2: Compilation completed successfully

// DEV1_SEQ_A: Generating NPU instructions for device: device1
// DEV1_SEQ_A: Generating NPU instructions for sequence: seq_a
// DEV1_SEQ_A: Compilation completed successfully

// ALL: Processing device: device1
// ALL: Processing device: device2
// ALL: Compilation completed successfully

// CTRLPKT_DEV1: Removing non-matching device: device2
// CTRLPKT_DEV1: Processing device: device1
// CTRLPKT_DEV1: Generating control packets for device: device1
// CTRLPKT_DEV1-NOT: Processing device: device2
// CTRLPKT_DEV1: Compilation completed successfully

// seq_a: 6 host args -> bo0..bo5, no bo6.
// COUNT_A: "name": "bo5"
// COUNT_A-NOT: "name": "bo6"

// seq_b: 7 host args -> bo0..bo6, no bo7.
// COUNT_B: "name": "bo6"
// COUNT_B-NOT: "name": "bo7"

// device1 uses column 0, device2 uses column 1 so that both devices can be
// compiled in the same pass manager invocation without tile address conflicts.
module {
  aie.device(npu1_2col) @device1 {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)

    aie.objectfifo @in1(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core = aie.core(%tile02) {
      aie.end
    }

    // seq_a and seq_b declare a DIFFERENT number of host arguments (6 vs 7,
    // both above the kMinHostBOs=5 floor so the counts are distinguishable).
    // This guards that --sequence-name drives the kernels.json boN count from
    // the SELECTED sequence, not simply the first one on the device.
    aie.runtime_sequence @seq_a(%arg0 : memref<16xi32>, %arg1 : memref<16xi32>, %arg2 : memref<16xi32>, %arg3 : memref<16xi32>, %arg4 : memref<16xi32>, %arg5 : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%arg1[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @out1, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%arg0[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @in1, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out1}
    }

    aie.runtime_sequence @seq_b(%arg0 : memref<16xi32>, %arg1 : memref<16xi32>, %arg2 : memref<16xi32>, %arg3 : memref<16xi32>, %arg4 : memref<16xi32>, %arg5 : memref<16xi32>, %arg6 : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%arg1[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @out1, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%arg0[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @in1, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out1}
    }
  }

  aie.device(npu1_2col) @device2 {
    %tile10 = aie.tile(1, 0)
    %tile12 = aie.tile(1, 2)

    aie.objectfifo @in2(%tile10, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2(%tile12, {%tile10}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core = aie.core(%tile12) {
      aie.end
    }

    aie.runtime_sequence @seq_c(%arg0 : memref<16xi32>, %arg1 : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%arg1[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @out2, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%arg0[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @in2, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out2}
    }
  }
}
