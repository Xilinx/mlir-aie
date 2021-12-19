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
// RUN: clang++ -O2 --target=aie -c -I/usr/include/aie %S/kernel.cpp
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib%/ %aie_runtime_lib%/test_library.cpp %S/test.cpp %S/kernel.cpp -lstdc++ -o test.elf
// RUN: %run_on_board ./test.elf

module @test {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<32x32xi32>
  %buf13_1 = AIE.buffer(%tile13) { sym_name = "debuf" } : memref<32x32xf32>

  %lock13_3 = AIE.lock(%tile13, 3)

  func private @func(%A: memref<32x32xi32>, %MinRe : f32, %MaxRe : f32, %MinIm : f32, %MaxIm : f32) -> ()
  func private @do_line(%A: memref<32x32xi32>, %MinRe : f32, %StepRe : f32, %Im : f32, %cols : i32) -> ()

  %core13 = AIE.core(%tile13) {
    %MinRe = constant -1.5 : f32
    %MaxRe = constant 0.5 : f32
    %MinIm = constant -1.0 : f32
    %MaxIm = constant 1.0 : f32
    %size = constant 1024 : i32

    %Frac = constant 1024.0 : f32
    %DiffRe = std.subf %MaxRe, %MinRe : f32
    %StepRe = std.divf %DiffRe, %Frac : f32
    %DiffIm = std.subf %MaxIm, %MinIm : f32
    %StepIm = std.divf %DiffIm, %Frac : f32

    %lb = constant 0 : index
    %ub = constant 1024 : index
    %step = constant 1 : index
    %c0 = constant 0 : index

    %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%Im = %MinIm) -> (f32) {
      %Im_next = std.addf %Im, %StepIm : f32
      AIE.useLock(%lock13_3, "Acquire", 1, 1) // acquire
      // memref.store %Im, %buf13_1[%c0, %iv] : memref<32x32xf32>
       call @do_line(%buf13_0, %MinRe, %StepRe, %Im, %size) : (memref<32x32xi32>, f32, f32, f32, i32) -> ()
      AIE.useLock(%lock13_3, "Release", 0, 1) // release for write
      scf.yield %Im_next : f32
    }
    // AIE.useLock(%lock13_3, "Acquire", 1, 1) // acquire
    // call @func(%buf13_0, %MinRe, %MaxRe, %MinIm, %MaxIm) : (memref<32x32xi32>, f32, f32, f32, f32) -> ()
    // AIE.useLock(%lock13_3, "Release", 0, 1) // release for write
    // AIE.useLock(%lock13_3, "Acquire", 1, 1) // acquire
    // call @func(%buf13_0, %MinRe, %center, %MinIm, %center) : (memref<32x32xi32>, f32, f32, f32, f32) -> ()
    // AIE.useLock(%lock13_3, "Release", 0, 1) // release for write
    // AIE.useLock(%lock13_3, "Acquire", 1, 1) // acquire
    // call @func(%buf13_0, %center, %MaxRe, %MinIm, %center) : (memref<32x32xi32>, f32, f32, f32, f32) -> ()
    // AIE.useLock(%lock13_3, "Release", 0, 1) // release for write
    // AIE.useLock(%lock13_3, "Acquire", 1, 1) // acquire
    // call @func(%buf13_0, %MinRe, %center, %center, %MaxIm) : (memref<32x32xi32>, f32, f32, f32, f32) -> ()
    // AIE.useLock(%lock13_3, "Release", 0, 1) // release for write
    // AIE.useLock(%lock13_3, "Acquire", 1, 1) // acquire
    // call @func(%buf13_0, %center, %MaxRe, %center, %MaxIm) : (memref<32x32xi32>, f32, f32, f32, f32) -> ()
    // AIE.useLock(%lock13_3, "Release", 0, 1) // release for write
    AIE.end
  } { link_with="kernel.o" }
}
