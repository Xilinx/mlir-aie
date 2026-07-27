//===- hrx_xclbin_sections.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Package a real .xclbin with the bundled XRT-free hrx-xclbinutil
// (built with -DAIE_BUILD_HRXXCLBINUTIL=ON, installed as `xclbinutil` in the
// AIE tools dir), then confirm the container actually carries the NPU sections
// aiecc and the amdxdna HAL rely on. This in-tree, hardware-free "section check"
// guards against the trimmed xclbinutil dropping a section handler or aiecc's
// packaging flags drifting.
//
// Gated on the `hrxxclbinutil` feature so it only runs when that tool was built;
// the tools dir is first on PATH, so the bare `xclbinutil` below is the bundled
// one (not a system XRT copy).

// REQUIRES: peano
// REQUIRES: hrxxclbinutil

// RUN: %aiecc --no-xchesscc --get-xclbin --xclbin-name=hrx_sections.xclbin %s
// RUN: xclbinutil --info --input hrx_sections.xclbin | FileCheck %s

// CHECK-DAG: MEM_TOPOLOGY
// CHECK-DAG: AIE_PARTITION
// CHECK-DAG: EMBEDDED_METADATA

module {
  aie.device(npu1) {
  %12 = aie.tile(1, 2)
  %buf = aie.buffer(%12) : memref<256xi32>
  %4 = aie.core(%12)  {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    memref.store %0, %buf[%1] : memref<256xi32>
    aie.end
  }
  }
}
