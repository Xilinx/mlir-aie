// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-device/per-sequence native builders must neither include other sequences
// nor mutate the shared input consumed by the parallel static-binary branch.
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aiecc --get-npu-cpp --get-npu-insts --output-dir=%t.d --tmpdir=%t.d/work %s
// RUN: %host_clang -std=c++17 -I%S/../../include -DGEN_HDR='"%t.d/npu_first_one.cpp"' -DGEN_FN=generate_txn_first_one %S/Inputs/npu_cpp_dump.cpp %host_link_flags -o %t.d/one
// RUN: %t.d/one %t.d/one.bin && cmp %t.d/one.bin %t.d/insts_first_one.bin
// RUN: %host_clang -std=c++17 -I%S/../../include -DGEN_HDR='"%t.d/npu_first_two.cpp"' -DGEN_FN=generate_txn_first_two %S/Inputs/npu_cpp_dump.cpp %host_link_flags -o %t.d/two
// RUN: %t.d/two %t.d/two.bin && cmp %t.d/two.bin %t.d/insts_first_two.bin
// RUN: %host_clang -std=c++17 -I%S/../../include -DGEN_HDR='"%t.d/npu_second_one.cpp"' -DGEN_FN=generate_txn_second_one %S/Inputs/npu_cpp_dump.cpp %host_link_flags -o %t.d/second
// RUN: %t.d/second %t.d/second.bin && cmp %t.d/second.bin %t.d/insts_second_one.bin
// RUN: FileCheck %s --check-prefix=FIRST --input-file=%t.d/npu_first_one.cpp
// RUN: FileCheck %s --check-prefix=SECOND --input-file=%t.d/npu_second_one.cpp
// RUN: aiecc --get-npu-cpp --npu-cpp-name=%t.d/selected.cpp --npu-cpp-emit-dispatch-shim --device-name=first --sequence-name=two --tmpdir=%t.d/filter %s
// RUN: FileCheck %s --check-prefix=FILTER --input-file=%t.d/selected.cpp
// RUN: %host_clang -std=c++17 -I%S/../../include -DGEN_HDR='"%t.d/selected.cpp"' -DGEN_FN=generate_txn_first_two -DCHECK_SHIM -DABI_STRING='""' %S/Inputs/npu_cpp_dump.cpp %host_link_flags -o %t.d/selected
// RUN: %t.d/selected %t.d/selected.bin && cmp %t.d/selected.bin %t.d/insts_first_two.bin

// FIRST-NOT: generate_txn_second
// FIRST-NOT: generate_txn_first_two
// FIRST: generate_txn_first_one(
// FIRST-NOT: generate_txn_first_two
// FIRST-NOT: generate_txn_second
// FIRST-NOT: dispatch_generate
// SECOND-NOT: generate_txn_first
// SECOND: generate_txn_second_one(
// SECOND-NOT: generate_txn_first
// FILTER-NOT: generate_txn_first_one
// FILTER-NOT: generate_txn_second
// FILTER: generate_txn_first_two(
// FILTER: dispatch_abi(
// FILTER: dispatch_generate(
// FILTER-NOT: generate_txn_first_one
// FILTER-NOT: generate_txn_second

module {
  aie.device(npu2) @first {
    aie.runtime_sequence @one() {
      %address = arith.constant 256 : i32
      %value = arith.constant 17 : i32
      aiex.npu.write32(%address, %value) : i32, i32
      aiex.npu.preempt {level = 3 : ui8}
      aiex.npu.load_pdi {id = 7 : i32, size = 4096 : i32, address = 0xabc12345678 : ui64}
    }
    aie.runtime_sequence @two() {
      %address = arith.constant 260 : i32
      %value = arith.constant 29 : i32
      aiex.npu.write32(%address, %value) : i32, i32
    }
  }
  aie.device(npu1_1col) @second {
    aie.runtime_sequence @one() {
      %address = arith.constant 264 : i32
      %value = arith.constant 31 : i32
      aiex.npu.write32(%address, %value) : i32, i32
    }
  }
}
