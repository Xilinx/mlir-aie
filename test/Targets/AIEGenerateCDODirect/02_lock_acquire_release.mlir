// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin
// CHECK: Generating: {{.*}}aie_cdo_init.bin

module @test02_lock_acquire_release {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %a = aie.buffer(%tile_1_3) {sym_name = "a"} : memref<256xi32>
    %lock1 = aie.lock(%tile_1_3, 3) {sym_name = "lock1"}
    %lock2 = aie.lock(%tile_1_3, 5) {sym_name = "lock2"}
    %core_1_3 = aie.core(%tile_1_3) {
      aie.use_lock(%lock1, Acquire, 0)
      aie.use_lock(%lock2, Acquire, 0)
      aie.use_lock(%lock2, Release, 1)
      aie.end
    }
  }
}

