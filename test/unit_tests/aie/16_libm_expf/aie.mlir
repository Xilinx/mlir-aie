//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: !hsa
// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  %buf_a = aie.buffer(%tile13) { sym_name = "a" } : memref<256xf32>
  %buf_b = aie.buffer(%tile13) { sym_name = "b" } : memref<256xf32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "inout_lock" }

  func.func private @func(%A: memref<256xf32>, %B: memref<256xf32>) -> ()

  %core13 = aie.core(%tile13) {
    aie.use_lock(%lock13_3, "Acquire", 1) // acquire
    affine.for %arg0 = 0 to 256 {
      %val1 = affine.load %buf_a[%arg0] : memref<256xf32>
      %val2 = math.exp %val1 : f32
      affine.store %val2, %buf_b[%arg0] : memref<256xf32>
    }
    aie.use_lock(%lock13_3, "Release", 0) // release for write
    aie.end
  }
}
}
