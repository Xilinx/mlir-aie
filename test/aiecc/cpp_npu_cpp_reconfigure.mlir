// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Materialize configure/run before C++ translation, then assign load-PDI IDs
// in module order, including devices outside the output filter.
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aiecc --get-npu-cpp --get-npu-insts --get=npu_lowered.mlir --device-name=main --sequence-name=sequence --output-dir=%t.d --tmpdir=%t.d/work %s
// RUN: FileCheck %s --check-prefix=LOWERED --input-file=%t.d/npu_lowered.mlir
// RUN: %host_clang -std=c++17 -I%S/../../include -DGEN_HDR='"%t.d/npu_main_sequence.cpp"' -DGEN_FN=generate_txn_main_sequence %S/Inputs/npu_cpp_dump.cpp %host_link_flags -o %t.d/native
// RUN: %t.d/native %t.d/native.bin && cmp %t.d/native.bin %t.d/insts_main_sequence.bin
// RUN: aiecc --get-npu-cpp --device-name=main --sequence-name=dynamic --output-dir=%t.d --tmpdir=%t.d/dynamic %s
// RUN: %host_clang -std=c++17 -I%S/../../include -DGEN_HDR='"%t.d/npu_main_dynamic.cpp"' -DGEN_FN=generate_txn_main_dynamic -DSCALAR %S/Inputs/npu_cpp_dump.cpp %host_link_flags -o %t.d/dynamic-native
// RUN: %t.d/dynamic-native %t.d/dynamic.bin 37 && cmp %t.d/dynamic.bin %t.d/insts_main_sequence.bin
// The real design also leaves dead memref view/cast chains after DMA lowering.
// RUN: aiecc --get-npu-cpp --get-npu-insts --device-name=main --sequence-name=sequence --output-dir=%t.d/views --tmpdir=%t.d/views/work %S/../npu-xrt/reconfigure_loadpdi/aie.mlir
// RUN: %host_clang -std=c++17 -I%S/../../include -DGEN_HDR='"%t.d/views/npu_main_sequence.cpp"' -DGEN_FN=generate_txn_main_sequence %S/Inputs/npu_cpp_dump.cpp %host_link_flags -o %t.d/views/native
// RUN: %t.d/views/native %t.d/views/native.bin && cmp %t.d/views/native.bin %t.d/views/insts_main_sequence.bin

// LOWERED-LABEL: aie.runtime_sequence()
// LOWERED: aiex.npu.load_pdi {device_ref = @config, id = 3 : i32}
// LOWERED: aiex.npu.write32
// LOWERED-NOT: aiex.configure
// LOWERED-NOT: aiex.run

module {
  aie.device(npu2) @unused {}
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence() {
      %value = arith.constant 37 : i32
      aiex.configure @config {
        aiex.run @write(%value) : (i32)
      }
    }
    aie.runtime_sequence @dynamic(%value: i32) {
      aiex.configure @config {
        aiex.run @write(%value) : (i32)
      }
    }
  }
  aie.device(npu2) @config {
    aie.runtime_sequence @write(%value: i32) {
      %address = arith.constant 256 : i32
      aiex.npu.write32(%address, %value) : i32, i32
    }
  }
}
