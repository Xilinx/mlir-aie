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

// C++-specific options must not silently do nothing for a non-C++ request.
// RUN: not aiecc --get-npu-insts --npu-cpp-emit-dispatch-shim --emit-dot 2>&1 | FileCheck %s --check-prefix=SHIM-REQUIRES-CPP
// RUN: not aiecc --npu-cpp-name=unused.cpp --emit-dot 2>&1 | FileCheck %s --check-prefix=NAME-REQUIRES-CPP
// RUN: not aiecc --get=npu_lowered.mlir --npu-cpp-name=unused.cpp --emit-dot 2>&1 | FileCheck %s --check-prefix=NAME-REQUIRES-CPP
// RUN: aiecc --get-npu-insts --npu-cpp-emit-dispatch-shim=false --emit-dot > %t.d/no-shim.dot
// SHIM-REQUIRES-CPP: aiecc: --npu-cpp-emit-dispatch-shim requires NPU C++ output; use --get-npu-cpp
// NAME-REQUIRES-CPP: aiecc: --npu-cpp-name requires NPU C++ output; use --get-npu-cpp

// Exact edge selectors, including custom names, are alternatives to the shorthand.
// RUN: aiecc --get=npu_{0}.cpp --npu-cpp-emit-dispatch-shim --device-name=first --sequence-name=two --output-dir=%t.d/exact --tmpdir=%t.d/exact/work %s
// RUN: FileCheck %s --check-prefix=FILTER --input-file=%t.d/exact/npu_first_two.cpp
// RUN: aiecc -g=custom.cpp --npu-cpp-name=custom.cpp --npu-cpp-emit-dispatch-shim --device-name=first --sequence-name=two --output-dir=%t.d/custom --tmpdir=%t.d/custom/work %s
// RUN: FileCheck %s --check-prefix=FILTER --input-file=%t.d/custom/custom.cpp

// Checkpoint cuts can select C++ without a final output. A resume may select a
// different artifact while retaining the original graph's C++ options.
// RUN: aiecc --cut=custom.cpp --checkpoint=%t.d/checkpoint --npu-cpp-name=custom.cpp --npu-cpp-emit-dispatch-shim --device-name=first --sequence-name=two --output-dir=%t.d/cut --tmpdir=%t.d/cut/work %s
// RUN: aiecc --resume=%t.d/checkpoint/manifest.json --get=custom.cpp
// RUN: FileCheck %s --check-prefix=FILTER --input-file=%t.d/cut/custom.cpp
// RUN: aiecc --resume=%t.d/checkpoint/manifest.json --get=insts_{0}.bin
// RUN: cmp %t.d/cut/insts_first_two.bin %t.d/insts_first_two.bin

// Filename templates need only distinguish the selected outputs; a fixed name
// works with the filter above, but multiple items or output kinds must not clash.
// RUN: not aiecc --get-npu-cpp --npu-cpp-name=same.cpp --output-dir=%t.d/duplicate --tmpdir=%t.d/duplicate/work %s 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not aiecc --get-npu-cpp --get-npu-insts --npu-cpp-name=insts_{0}.bin --output-dir=%t.d/collision --tmpdir=%t.d/collision/work %s 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// DUPLICATE: produced duplicate output path

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
