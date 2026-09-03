//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %s --aie-npu-to-cpp --aie-npu-emit-dispatch-shim | FileCheck %s
// RUN: aie-translate %s --aie-npu-to-cpp | FileCheck %s --check-prefix=NOSHIM
// RUN: aie-translate %S/Inputs/dispatch_shim_widths.mlir --aie-npu-to-cpp \
// RUN:   --aie-npu-emit-dispatch-shim | FileCheck %s --check-prefix=SIGNED

// Both symbols are built from the same aie.runtime_sequence argument types, so
// the names dispatch_abi() reports must match dispatch_generate()'s signature
// exactly -- Python builds its ctypes call signature from that pairing.

// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int32_t {{v[0-9]+}}, size_t {{v[0-9]+}})

// CHECK: extern "C" const char* dispatch_abi() {
// CHECK-NEXT: return "int32_t,size_t";

// CHECK: extern "C" int64_t dispatch_generate(int32_t [[A:v[0-9]+]], size_t [[B:v[0-9]+]], uint32_t** [[OUT:v[0-9]+]]) {
// CHECK-NEXT: static thread_local std::vector<uint32_t> __txn;
// CHECK-NEXT: auto __result = generate_txn_main_seq([[A]], [[B]]);
// The builder declines (std::nullopt) when a runtime scalar overflows a BD
// field; -2 is the sentinel DispatchBridge turns into an exception.
// CHECK-NEXT: if (!__result) return -2;
// CHECK-NEXT: __txn = std::move(*__result);
// CHECK-NEXT: *[[OUT]] = __txn.data();
// CHECK-NEXT: return static_cast<int64_t>(__txn.size());

// Off by default: these are definitions, so a header including them twice
// would not link.
// NOSHIM-NOT: dispatch_abi
// NOSHIM-NOT: dispatch_generate

// The ABI string has to track CppEmitter's signedness and width rules: naming
// int32_t for a uint32_t parameter would build a signed ctypes signature
// against an unsigned entry point.
// SIGNED: generate_txn_main_widths(uint32_t {{v[0-9]+}}, bool {{v[0-9]+}}, int8_t {{v[0-9]+}}, size_t {{v[0-9]+}})
// SIGNED: return "uint32_t,bool,int8_t,size_t";

module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %param: i32, %n: index) {
      aiex.npu.address_patch(%param : i32) {addr = 119300 : ui32, arg_idx = 2 : i32}
    }
  }
}
