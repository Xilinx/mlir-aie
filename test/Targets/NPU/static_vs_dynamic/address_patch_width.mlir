// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: mkdir -p %t.d
// RUN: aie-translate %s --aie-npu-to-cpp > %t.d/fold.h
// RUN: aie-translate %s --aie-npu-to-cpp --aie-npu-fold-ddr-addr-offset=false > %t.d/nofold.h
// RUN: aie-translate %s --aie-npu-to-binary -aie-output-binary=false -aie-sequence-name=static > %t.d/fold.hex
// RUN: aie-translate %s --aie-npu-to-binary -aie-output-binary=false -aie-sequence-name=static --aie-npu-fold-ddr-addr-offset=false > %t.d/nofold.hex
// RUN: %host_clang -std=c++17 -I%S/../../../../include -DGEN_HDR='"%t.d/fold.h"' -DSTATIC_FN=generate_txn_main_static -DDYN_FN=generate_txn_main_dynamic -DARGVAL=-2147483648 %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/fold.exe
// RUN: %t.d/fold.exe %t.d/fold.hex
// RUN: %host_clang -std=c++17 -I%S/../../../../include -DGEN_HDR='"%t.d/nofold.h"' -DSTATIC_FN=generate_txn_main_static -DDYN_FN=generate_txn_main_dynamic -DARGVAL=-2147483648 %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/nofold.exe
// RUN: %t.d/nofold.exe %t.d/nofold.hex

// Byte comparisons include both offset words, across the firmware-translated
// argument boundary, with the i32 sign bit set and with a genuine i64 offset.
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @static() {
      %min = arith.constant -2147483648 : i32
      %max = arith.constant -1 : i32
      %wide = arith.constant 8589934591 : i64
      aiex.npu.address_patch(%min : i32) {addr = 119300 : ui32, arg_idx = 4 : i32}
      aiex.npu.address_patch(%min : i32) {addr = 119300 : ui32, arg_idx = 5 : i32}
      aiex.npu.address_patch(%max : i32) {addr = 119300 : ui32, arg_idx = 4 : i32}
      aiex.npu.address_patch(%max : i32) {addr = 119300 : ui32, arg_idx = 5 : i32}
      aiex.npu.address_patch(%wide : i64) {addr = 119300 : ui32, arg_idx = 5 : i32}
    }
    aie.runtime_sequence @dynamic(%min: i32) {
      %mask = arith.constant 2147483647 : i32
      %max = arith.ori %min, %mask : i32
      %wide = arith.constant 8589934591 : i64
      aiex.npu.address_patch(%min : i32) {addr = 119300 : ui32, arg_idx = 4 : i32}
      aiex.npu.address_patch(%min : i32) {addr = 119300 : ui32, arg_idx = 5 : i32}
      aiex.npu.address_patch(%max : i32) {addr = 119300 : ui32, arg_idx = 4 : i32}
      aiex.npu.address_patch(%max : i32) {addr = 119300 : ui32, arg_idx = 5 : i32}
      aiex.npu.address_patch(%wide : i64) {addr = 119300 : ui32, arg_idx = 5 : i32}
    }
  }
}
