//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static oracle for the NOT-TAKEN branch of the rolled dynamic scf.if
// (rolled_if.mlir): an empty runtime sequence. When the runtime condition is
// false the dynamic builder emits no BD configuration, so its register footprint
// must match this no-op. Companion input under Inputs/ so lit ignores it.
//
//===----------------------------------------------------------------------===//

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @static_nottaken(%arg0: memref<1024xi32>) {
  }
}
