<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->
# <ins>Tutorial 2b - Simulation</ins>

In [tutorial-2a](../tutorial-2a), we described how to construct the host code `test.cpp` file for configuring, running and testing our AI Engine design. This tutorial focuses on how to run a hybrid software simulation of our design with the Vitis aiesimulator. While single kernel simulation is important and covered in [tutorial-9](../../tutorial-9), we focus here on simulating our entire AI Engine system design where individual tiles run a cycle accurate simulation and communication between tiles are simulated at the transaction level. 

By default, `aiecc.py` will generate the `Work` directory which contains the stub and configuration files necessary for integrating with aiesimulator. In our [Makefile](./Makefile), we call `make` to build the simulator object file.
```
make -C Work/ps/c_rts/systemC link
```
> Note: Make sure the environment variable `MLIR_AIE_DIR` is set to the absolute path to your mlir-aie repo so the simulator can find the necessary reference files. 

Aiesimulator is then invoked with the following command:
```
aiesimulator --pkg-dir=./Work --dump-vcd foo
```
This command points the simulator to the local `Work` directory and dumps the simulation waveform vcd to foo.vcd. The simulation results will be outputted to the terminal as applicable. 

Because aiesimulator is to set to run a cycle accurate simulation for each tile, the simulation time for our small tutorial design only take a few minutes to complete on a standard machine. However, much larger designs that involve a large number of tiles may take much longer. A more effective strategy would be to simulate a portion of that design in aiesimulator and run the full design directly on hardware.

> Some things to note about differences between running simulation and running on hardware. It is important to note the differences between host code timing and AIE timing in simulation compared to hardware. In hardware, the AI engines complete operation extremely quickly so host code commands take a much longer time compared to AI Engine program cycles. As a result, a host command like usleep could be used to wait for a program to be done when AIE operations complete very quickly. However, in simulation, the opposite is true which is why we use host API functions like `mlir_aie_acquire_lock` with a timeout value to ensure we are synchronizing AIE simulation and host code timings. These differences can also come up in unique ways which we will highlight in later tutorials.

## <ins>Tutorial 2b Lab</ins>

1. Run the aiesimulator via make.
    ```
    make sim
    ```
    You should see the simulator print a number o aiesimulator related status messages before finally running the host code and outputting the `PASS` message.

2. Modify the host code `test.cpp` to add a `mlir_aie_print_tile_status` for tile(1,4) and rerun the simulator.


The next [tutorial-2c](../tutorial-2c) walks us through running our design on hardware and measuring performance.