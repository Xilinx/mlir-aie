//===- passthrough_dynamic.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Dynamic counterpart of `cpp_static_vs_dynamic_txn.mlir`.
//
// Mirrors the static module structurally; the only difference is that the
// runtime_sequence takes an additional %n : i32 argument that is forwarded
// to the `aiex.npu.rtp_write` operations.
//
// All DMA descriptor fields (sizes, strides, offsets) are kept as
// constants on both sides so the BD-lowering pass folds them into a
// `blockwrite` TXN op on each side; otherwise, threading %n through the
// BD would force the compiler to emit per-register `write32` ops on the
// dynamic side and the two streams would differ structurally even when
// they program the same hardware state.
//
// Calling generate_txn_sequence(4096) on this header must produce the
// exact same word stream that generate_txn_sequence() in the static
// header produces.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    %rtp = aie.buffer(%tile_0_2) {sym_name = "rtp"} : memref<16xi32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c1_i32 = arith.constant 1 : i32

      %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

      %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

      scf.for %i = %c0 to %c64 step %c1 {
        %val = memref.load %elem_in[%i] : memref<64xi32>
        %result = arith.addi %val, %c1_i32 : i32
        memref.store %result, %elem_out[%i] : memref<64xi32>
      }

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } {link_with = ""}

    aie.runtime_sequence(%in : memref<8192xi32>, %out : memref<8192xi32>, %n : i32) {
      // %n drives the first rtp_write directly and a derived value drives
      // the second.  Both go through txn_append_write32, where the encoded
      // word equals the unsigned cast of the runtime value — bit-identical
      // to the static version when %n == the static constant.
      %c1_i32 = arith.constant 1 : i32
      %n_plus_1 = arith.addi %n, %c1_i32 : i32

      aiex.npu.rtp_write(@rtp, 0 : ui32, %n)        : i32
      aiex.npu.rtp_write(@rtp, 4 : ui32, %n_plus_1) : i32

      aiex.npu.write32 {address = 196612 : ui32, value = 42 : ui32}

      // First 4-D pattern: sizes=[2,4,8,64], strides=[2048,512,64,1] over %out.
      // dma_memcpy_nd offsets/sizes/strides are i64 SSA values (TODO: i32).
      %c0   = arith.constant    0 : i64
      %c1   = arith.constant    1 : i64
      %c2   = arith.constant    2 : i64
      %c4   = arith.constant    4 : i64
      %c8   = arith.constant    8 : i64
      %c16  = arith.constant   16 : i64
      %c32  = arith.constant   32 : i64
      %c64  = arith.constant   64 : i64
      %c128 = arith.constant  128 : i64
      %c256 = arith.constant  256 : i64
      %c512 = arith.constant  512 : i64
      %c2048 = arith.constant 2048 : i64
      %c4096 = arith.constant 4096 : i64

      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0]
                                 [%c2,%c4,%c8,%c64]
                                 [%c2048,%c512,%c64,%c1])
        {metadata = @of_out, id = 1 : i64} : memref<8192xi32>

      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0]
                                [%c2,%c4,%c8,%c64]
                                [%c2048,%c512,%c64,%c1])
        {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<8192xi32>

      aiex.npu.dma_wait {symbol = @of_out}

      // Second 4-D pattern with a different shape/stride mix: sizes=[1,8,16,32],
      // strides=[4096,512,32,1].  This exercises a fresh BD blockwrite + the
      // address_patch for a different arg index, touching more TXN words.
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0]
                                 [%c1,%c8,%c16,%c32]
                                 [%c4096,%c512,%c32,%c1])
        {metadata = @of_out, id = 1 : i64} : memref<8192xi32>

      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0]
                                [%c1,%c8,%c16,%c32]
                                [%c4096,%c512,%c32,%c1])
        {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<8192xi32>

      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
