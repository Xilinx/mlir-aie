//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano
// RUN: %PEANO_INSTALL_DIR/bin/clang++ -O2 --target=aie -c -I/usr/include/aie %S/kernel.cpp
// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_libg %S/test.cpp %S/kernel.cpp -lstdc++ -o test.elf
// RUN: %run_on_board ./test.elf

module @test {
  %tile13 = aie.tile(1, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<32x32xi32>
  %buf13_1 = aie.buffer(%tile13) { sym_name = "debuf" } : memref<32x32xf32>

  %lock13_3 = aie.lock(%tile13, 3)

  func.func private @func(%A: memref<32x32xi32>, %MinRe : f32, %MaxRe : f32, %MinIm : f32, %MaxIm : f32) -> ()
  func.func private @do_line(%A: memref<32x32xi32>, %MinRe : f32, %StepRe : f32, %Im : f32, %cols : i32) -> ()

  %core13 = aie.core(%tile13) {
    %MinRe = arith.constant -1.5 : f32
    %MaxRe = arith.constant 0.5 : f32
    %MinIm = arith.constant -1.0 : f32
    %MaxIm = arith.constant 1.0 : f32
    %size = arith.constant 1024 : i32

    %Frac = arith.constant 1024.0 : f32
    %DiffRe = arith.subf %MaxRe, %MinRe : f32
    %StepRe = arith.divf %DiffRe, %Frac : f32
    %DiffIm = arith.subf %MaxIm, %MinIm : f32
    %StepIm = arith.divf %DiffIm, %Frac : f32

    %lb = arith.constant 0 : index
    %ub = arith.constant 1024 : index
    %step = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%Im = %MinIm) -> (f32) {
      %Im_next = arith.addf %Im, %StepIm : f32
      aie.use_lock(%lock13_3, "Acquire", 1) // acquire
      func.call @do_line(%buf13_0, %MinRe, %StepRe, %Im, %size) : (memref<32x32xi32>, f32, f32, f32, i32) -> ()
      aie.use_lock(%lock13_3, "Release", 0) // release for write
      scf.yield %Im_next : f32
    }
    aie.end
  } { link_with="kernel.o" }
}
