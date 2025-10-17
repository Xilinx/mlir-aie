<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->
# <ins>Tutorial 11 - Scaling up to large multi-core designs</ins>

To scale up to large multi-core designs, we need to consider how such designs are built and how they would be verified.

## <ins>Building large designs</ins>
This topic is an active area of research and one of the things that the MLIR-AIE dialect enables is for tools to be build to help build these large designs. One example of this methodology can be seen in the [reference design/prime_sieve_large](../../reference_designs/prime_sieve_large). Here, we use an python script to stamp out the design into a large `aie.mlir` design. While this is certainly one way to do it, we can also take the functions that the python script performs and integrate them as part of an MLIR pass which might take an IR representation of our stamped out design and generate the MLIR-AIE representation. 

## <ins>Verifying large designs</ins>
The second topic surrounding large designs involves how to verify their functionality. Simulations of large designs, while possible, may become infeasible with cycle accurate simulators for every AI Engine tile. Running these on a board would drastically reduce the simulation time as we would be running the design in real time. Alternatively, a good strategy may involve verifying components of the design as a smaller 2x2 sub-sample in simulation prior to stamping out the sub-sample over the entire array.

## <ins>Tutorial-11 Lab</ins>
1. Run the python script [code_gen.py](../../reference_designs/prime_sieve_large/code_gen.py) to rebuild the `aie.mlir` source. Then run `aiecc.py` based on what you've already learned to build the .elf and .exe files to run this design on the board. Run the design on the board.
