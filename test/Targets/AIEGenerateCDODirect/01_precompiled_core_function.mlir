// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @test_chesss_01_precompiled_core_function {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %a = aie.buffer(%tile_1_3) {sym_name = "a"} : memref<256xi32>
    %b = aie.buffer(%tile_1_3) {sym_name = "b"} : memref<256xi32>
    %input_lock = aie.lock(%tile_1_3, 3) {sym_name = "input_lock"}
    %output_lock = aie.lock(%tile_1_3, 5) {sym_name = "output_lock"}
    // func.func private @func(memref<256xi32>, memref<256xi32>)
    %core_1_3 = aie.core(%tile_1_3) {
      aie.use_lock(%input_lock, Acquire, 1)
      aie.use_lock(%output_lock, Acquire, 0)
      // func.call @func(%a, %b) : (memref<256xi32>, memref<256xi32>) -> ()
      aie.use_lock(%input_lock, Release, 0)
      aie.use_lock(%output_lock, Release, 1)
      aie.end
    } // {link_with = "kernel.o"}
  }
}

