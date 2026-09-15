//===- bank_placement_mismatch.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A core that places its two lookup tables in separate banks, and a linker that
// packs them into one.
//
// `aie::lut<4>` reads the pair in parallel, so the placement is what makes the
// two reads independent. The compiler schedules on it and nothing downstream
// re-checks it, so an unsatisfied request reads the wrong bank at full speed
// and corrupts results with no diagnostic. aiecc compares the request in the
// objects against the address in the linked ELF and fails the build.
//
// Requests are read from the input objects, not the linked ELF: the chess
// linker merges `.bss.DM_bankB` into `.bss.DM_bankA`, so the linked section
// names describe a grouping rather than a request.

// REQUIRES: chess
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: xchesscc_wrapper aie2p -c %S/bank_placement_mismatch_kernel.cc -o %t.d/bank_placement_mismatch_kernel.o
// RUN: cd %t.d && not aiecc --xchesscc --xbridge --get-core-elfs %s 2>&1 | FileCheck %s

// The table that asked for bank B is the one reported; the bank-A table got
// the bank it asked for and must not be named.
// CHECK: error: core (0, 2): 'activation_lut_cd' is placed for memory bank B
// CHECK-SAME: but the linker put it at 0x{{[0-9a-f]+}}, which is bank A
// CHECK-NOT: 'activation_lut_ab' is placed

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    func.func private @classify(memref<64xi8>) attributes {link_with = "bank_placement_mismatch_kernel.o"}

    // A small stack, so every bank has room: the tables share one only because
    // nothing told the linker to spread them.
    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<64xi8>
      func.call @classify(%e) : (memref<64xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 1024 : i32 }

    aie.runtime_sequence(%out : memref<64xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
