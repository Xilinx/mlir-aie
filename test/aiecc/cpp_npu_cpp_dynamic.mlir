// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A C++-only request must not reach the static binary translator. Reusing one
// native builder at two scalar values produces the matching static streams.
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aiecc --get-npu-cpp --npu-cpp-emit-dispatch-shim --sequence-name=dynamic --get=npu_lowered.mlir --output-dir=%t.d --tmpdir=%t.d/work %s
// RUN: FileCheck %s --check-prefix=ABI --input-file=%t.d/npu_main_dynamic.cpp
// RUN: aiecc --get-npu-insts --sequence-name=static17 --output-dir=%t.d --tmpdir=%t.d/static17 %s
// RUN: aiecc --get-npu-insts --sequence-name=static29 --output-dir=%t.d --tmpdir=%t.d/static29 %s
// RUN: %host_clang -std=c++17 -I%S/../../include -DGEN_HDR='"%t.d/npu_main_dynamic.cpp"' -DGEN_FN=generate_txn_main_dynamic -DSCALAR -DCHECK_SHIM -DABI_STRING='"int32_t"' %S/Inputs/npu_cpp_dump.cpp %host_link_flags -o %t.d/dynamic
// RUN: %t.d/dynamic %t.d/17.bin 17 && cmp %t.d/17.bin %t.d/insts_main_static17.bin
// RUN: %t.d/dynamic %t.d/29.bin 29 && cmp %t.d/29.bin %t.d/insts_main_static29.bin
// RUN: aiecc --get-npu-cpp --sequence-name=dynamic --fold-ddr-addr-offset=false --output-dir=%t.d/unfolded --tmpdir=%t.d/unfolded/work %s
// RUN: aiecc --get-npu-insts --sequence-name=static17 --fold-ddr-addr-offset=false --output-dir=%t.d/unfolded --tmpdir=%t.d/unfolded/static17 %s
// RUN: %host_clang -std=c++17 -I%S/../../include -DGEN_HDR='"%t.d/unfolded/npu_main_dynamic.cpp"' -DGEN_FN=generate_txn_main_dynamic -DSCALAR %S/Inputs/npu_cpp_dump.cpp %host_link_flags -o %t.d/unfolded/dynamic
// RUN: %t.d/unfolded/dynamic %t.d/unfolded/17.bin 17 && cmp %t.d/unfolded/17.bin %t.d/unfolded/insts_main_static17.bin
// RUN: not cmp %t.d/17.bin %t.d/unfolded/17.bin

// ABI: generate_txn_main_dynamic(int32_t
// ABI: dispatch_generate(int32_t

module {
  aie.device(npu2) @main {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @input (%tile, MM2S, 0)
    aie.runtime_sequence @dynamic(%a: memref<64xi32>, %b: memref<64xi32>, %c: memref<64xi32>, %d: memref<64xi32>, %e: memref<64xi32>, %f: memref<64xi32>, %value: i32) {
      %address = arith.constant 256 : i32
      aiex.npu.write32(%address, %value) : i32, i32
      aiex.npu.dma_memcpy_nd(%f[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @input} : memref<64xi32>
    }
    aie.runtime_sequence @static17(%a: memref<64xi32>, %b: memref<64xi32>, %c: memref<64xi32>, %d: memref<64xi32>, %e: memref<64xi32>, %f: memref<64xi32>) {
      %address = arith.constant 256 : i32
      %value = arith.constant 17 : i32
      aiex.npu.write32(%address, %value) : i32, i32
      aiex.npu.dma_memcpy_nd(%f[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @input} : memref<64xi32>
    }
    aie.runtime_sequence @static29(%a: memref<64xi32>, %b: memref<64xi32>, %c: memref<64xi32>, %d: memref<64xi32>, %e: memref<64xi32>, %f: memref<64xi32>) {
      %address = arith.constant 256 : i32
      %value = arith.constant 29 : i32
      aiex.npu.write32(%address, %value) : i32, i32
      aiex.npu.dma_memcpy_nd(%f[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @input} : memref<64xi32>
    }
  }
}
