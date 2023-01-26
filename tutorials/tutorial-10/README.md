<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->
# <ins>Tutorial 10 - MLIR-AIE commands and utilities</ins>

The MLIR-AIE dialect builds a number of command utilities for compiling and transforming operations written in the MLIR-AIE dialect into other intermediate representations (IRs) as well as generating AIE tile elfs and host executables to be run on the board. The two main output utilities that building the MLIR-AIE project gives you is `aie-translate` adn `aie-opt` which are used to transform MLIR-AIE dialect into other IRs. These utilities are then used by the convenience python utility `aiecc.py` to compile operations written in MLIR-AIE dialect into elf/host executables.

The basic way that we use `aiecc.py` to compile our MLIR-AIE written code (e.g. aie.mlir) into elf/host executable (core_*.elf, test.exe) is the following:
```
aiecc.py -j4 --sysroot=<platform sysroot> -host-target=aarch64-linux-gnu aie.mlir -I<runtime_lib> <runtime lib>/test_library.cpp ./test.cpp -o test.exe
```
This command compiles the our source text file aie.mlir and the testbench/host code test.cpp to compile and generate the output host executable test.exe. The AIE tile elfs are generated automatically for each AIE tile that needs to be configured. Additionally, we pass references to -I<runtime lib> and compile the <runtime lib>/test_library.cpp as the common test functions are often used in the host code to initialize the AIE system design and read and write to components in the AIE array.

***TODO: more write-up to define use of aie-translate and aie-opt***
    