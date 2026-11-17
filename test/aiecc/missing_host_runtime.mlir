//===- missing_host_runtime.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc compiles a host program and the simulator model against the host
// runtime in <install prefix>/runtime_lib/<host arch>/, where the install
// prefix is the parent of the directory holding the aiecc executable. That
// runtime is built for each entry of AIE_RUNTIME_TARGETS, so an install made
// without the host's architecture there lacks it. Copying aiecc into a private
// prefix whose runtime_lib is empty reproduces that, so no toolchain is needed:
// the diagnostic fires before the host compiler would be launched.

// The test copies the aiecc executable. The prefix is removed before each run,
// not after: on NFS a just-executed binary can still be referenced for a few
// seconds, so removing its directory at the end of the test fails.
// UNSUPPORTED: system-windows

// RUN: rm -rf missing_host_runtime_prefix
// RUN: mkdir -p missing_host_runtime_prefix/bin missing_host_runtime_prefix/runtime_lib missing_host_runtime_prefix/aie_runtime_lib/AIE2/aiesim missing_host_runtime_prefix/no-aietools
// RUN: echo "int main(){return 0;}" > missing_host_runtime_prefix/host.cpp
// RUN: touch missing_host_runtime_prefix/aie_runtime_lib/AIE2/aiesim/genwrapper_for_ps.cpp
// RUN: cp %aiecc missing_host_runtime_prefix/bin/aiecc
// RUN: not ./missing_host_runtime_prefix/bin/aiecc %s --get-host --output-dir=missing_host_runtime_prefix/out --tmpdir=missing_host_runtime_prefix/work -- missing_host_runtime_prefix/host.cpp 2>&1 | FileCheck %s
// RUN: not ./missing_host_runtime_prefix/bin/aiecc %s --get-host --host-target=aarch64-linux-gnu --output-dir=missing_host_runtime_prefix/out --tmpdir=missing_host_runtime_prefix/work -- missing_host_runtime_prefix/host.cpp 2>&1 | FileCheck %s --check-prefix=AARCH64
// RUN: not ./missing_host_runtime_prefix/bin/aiecc %s --xchesscc --xbridge --get-aiesim --aietools=missing_host_runtime_prefix/no-aietools --output-dir=missing_host_runtime_prefix/sim --tmpdir=missing_host_runtime_prefix/sim-work 2>&1 | FileCheck %s --check-prefix=SIM
// RUN: mkdir -p missing_host_runtime_prefix/runtime_lib/x86_64
// RUN: not ./missing_host_runtime_prefix/bin/aiecc %s --get-host --output-dir=missing_host_runtime_prefix/out --tmpdir=missing_host_runtime_prefix/work -- missing_host_runtime_prefix/host.cpp 2>&1 | FileCheck %s --check-prefix=ALLOC
// RUN: ./missing_host_runtime_prefix/bin/aiecc %s --get-host -n -v --output-dir=missing_host_runtime_prefix/dry --tmpdir=missing_host_runtime_prefix/dry-work -- missing_host_runtime_prefix/host.cpp 2>&1 | FileCheck %s --check-prefix=DRY

// The diagnostic names the missing path and the build option that produces
// it. A host program needs runtime_lib/<arch> and the allocator it links; a
// cross-compiled one may take the xaiengine headers from its sysroot.
// CHECK: aiecc: missing host runtime library: {{.*}}missing_host_runtime_prefix/runtime_lib/x86_64{{$}}
// CHECK: aiecc: the host runtime is built per target; rebuild and install with x86_64 in AIE_RUNTIME_TARGETS

// AARCH64: aiecc: missing host runtime library: {{.*}}missing_host_runtime_prefix/runtime_lib/aarch64{{$}}
// AARCH64: aiecc: the host runtime is built per target; rebuild and install with aarch64 in AIE_RUNTIME_TARGETS

// ALLOC: aiecc: missing host runtime library: {{.*}}missing_host_runtime_prefix/runtime_lib/x86_64/test_lib/lib/libmemory_allocator_ion.a
// ALLOC-NOT: xaiengine
// ALLOC: rebuild and install with x86_64 in AIE_RUNTIME_TARGETS

// The simulator model also needs the test library headers; the allocator it
// links is optional.
// SIM: aiecc: missing host runtime library: {{.*}}missing_host_runtime_prefix/runtime_lib/x86_64/xaiengine/include
// SIM: aiecc: missing host runtime library: {{.*}}missing_host_runtime_prefix/runtime_lib/x86_64/xaiengine/lib
// SIM: aiecc: missing host runtime library: {{.*}}missing_host_runtime_prefix/runtime_lib/x86_64/test_lib/include
// SIM-NOT: libmemory_allocator
// SIM: rebuild and install with x86_64 in AIE_RUNTIME_TARGETS

// A dry run never touches the file system, so it still prints the compile
// command.
// DRY-NOT: missing host runtime library
// DRY: clang++

module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
  }
}
