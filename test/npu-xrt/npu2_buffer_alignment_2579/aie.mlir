//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Regression test for mlir-aie issue #2579 on NPU2 (AIE2P).
//
// stack_size = 1028 is deliberately not a multiple of 64.  Pre-fix, the
// allocator placed the first compute-tile buffer (out1_buff_0) at address
// 1056 — 32-byte aligned but not 64-byte aligned — and the kernel's
// aie::load_v<int32_t,16> (a 512-bit load that requires 64B alignment on
// AIE2P) returned wrong data.  The fix overrides
// getComputeTileLoadStoreBusWidth() = 512 on BaseNPU2TargetModel so the
// allocator pads up to 64B and out1_buff_0 lands at 1088.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @in1(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out1(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>

    func.func private @alignment_test_kernel(memref<32xi32>, memref<32xi32>) attributes {link_with = "kernel.o"}

    aie.core(%tile_0_2) {
      %sub_in   = aie.objectfifo.acquire @in1(Consume, 1)  : !aie.objectfifosubview<memref<32xi32>>
      %elem_in  = aie.objectfifo.subview.access %sub_in[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
      %sub_out  = aie.objectfifo.acquire @out1(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
      %elem_out = aie.objectfifo.subview.access %sub_out[0]: !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>

      func.call @alignment_test_kernel(%elem_in, %elem_out) : (memref<32xi32>, memref<32xi32>) -> ()

      aie.objectfifo.release @in1(Consume, 1)
      aie.objectfifo.release @out1(Produce, 1)
      aie.end
    } {stack_size = 1028 : i32}

    aie.runtime_sequence(%in : memref<32xi32>, %out : memref<32xi32>) {
      %c0  = arith.constant 0 : i64
      %c1  = arith.constant 1 : i64
      %c32 = arith.constant 32 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c32][%c0,%c0,%c0,%c1]) {metadata = @out1, id = 1 : i64} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%in [%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c32][%c0,%c0,%c0,%c1]) {metadata = @in1,  id = 0 : i64, issue_token = true} : memref<32xi32>
      aiex.npu.dma_wait {symbol = @out1}
    }
  }
}
