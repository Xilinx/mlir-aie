<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->
# <ins>Tutorial 2 - Host code configuration, simulation, hardware performance</ins>

Now that we've been introduced to the 3 main components of AI Engine tile and have successfully compiled our design, lowering and transforming it from the MLIR-AIE dialect into a AI Engine program (.elf file), we want to take a look at how we can interact with this design in 3 main areas.

1. Initialize and configure the design (to run in simulation or on a board) in [tutorial-2a](./tutorial-2a)
2. Run a software simulation of the design in [tutorial-2b](./tutorial-2b)
3. Run the design in hardware and measure performance in [tutorial-2c](./tutorial-2c)

All of these topics allow us to interact with our design at a system level, to ensure correct functionality and execute our design in simulation and live hardware. They bring the generated individual tile programs you built in [tutorial-1](../tutorial-1) down to simulation and implementation on AMD hardware.

Let's start with initializing and configuring our design in [tutorial-2a](./tutorial-2a).
