//===- lut_runtime_lib_banks.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The runtime library's own exp tables land in separate banks.
//
// getExpBf16 builds two aie::lut<4> pairs out of exp_ilut_ab/_cd and
// exp_flut_ab/_cd. Before those definitions carried a bank request the linker
// packed each pair adjacently, into one bank, and the gather read the wrong
// port with nothing to report it. This is the case issue #3737 describes.
//
// data_size bounds the unpinned region, which would otherwise claim the run
// the pinned tables need.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I%S/../../third_party/aie_api/include -I%S/../../aie_runtime_lib/AIE2P -fembed-bitcode -c %S/lut_runtime_lib_banks_kernel.cc -o %t.d/lut_runtime_lib_banks_kernel.o
// RUN: cd %t.d && aiecc --get-core-elfs --check-lut-banks %s
// RUN: llvm-readelf -s %t.d/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s

// Local memory starts at 0x70000 and a bank is 0x4000, so an _ab in 0x70000-
// 0x73fff and its _cd in 0x74000-0x77fff are a bank apart.
// CHECK-DAG: 00071{{[0-9a-f]+}} {{.*}} exp_ilut_ab
// CHECK-DAG: 00074{{[0-9a-f]+}} {{.*}} exp_ilut_cd
// CHECK-DAG: 00071{{[0-9a-f]+}} {{.*}} exp_flut_ab
// CHECK-DAG: 00074{{[0-9a-f]+}} {{.*}} exp_flut_cd

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32xi8>>

    func.func private @classify(memref<32xi8>) attributes {link_with = "lut_runtime_lib_banks_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<32xi8>
      func.call @classify(%e) : (memref<32xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 1024 : i32, data_size = 4096 : i32 }

    aie.runtime_sequence(%out : memref<32xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c32 = arith.constant 32 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c32][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<32xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
