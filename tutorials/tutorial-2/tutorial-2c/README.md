<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->
# <ins>Tutorial 2c - Running on hardware and measuring performance</ins>

In [tutorial-2b](../tutorial-2b), we walked through the steps of running our design through aiesimulator. However, it may be easier and faster to run our design on hardware, particularly for very large designs. 

## <ins>Tutorial 2c Lab</ins>
1. Run `make` again in this directory.

You will have noticed from [tutorial-2a](../tutorial-2a) that `aiecc.py` generates a number of output files including the `core_1_4.elf` and `tutorial-2c.exe`. These files will be used now to run our design on hardware. In the `mlir-aie` repo, we have information on building a custom platform for the `vck190` that configures all the NOC connections to enable all shim DMAs to communicate to DDR memroy. Once your board is up and running, we can run our tutorial designs by copying our .elf and .exe files to the board. You can copy those files directly to the SD card or if the board is connected to your shared network, you can copy your files over with:
```
scp *elf xilinx@<board ip addr>:/home/xilinx/.
scp *exe xilinx@<board ip addr>:/home/xilinx/.
```
Then ssh onto the board and navigate to the directory where your design files are copied and execute the host code with sudo rights.
> username/password for a PYNQ based vck190 image: xilinx/xilinx
```
ssh xilinx@<board ip addr>
cd <location of copied elf/ exe files>
sudo ./tutorial-2c.exe
```
The expected output of a successful run should look like the following:
```
Tutorial-2c test start.
Start cores
Acquired lock14_0 (1) in tile (1,4). Done.
Checking buf[3] = 14.
PASS!
Tutorial-2c test done.
```

## <ins>Advanced Topic - Performance measurement in hardware</ins>
Now that we've compiled, simulated and run our design. We can take a step back to leverage some of the AI Engine's built in performance measurement hardware (timers and event traces) to capture actual on-board performance. 

We can first declare some host code variable to hold timer values.
```
u32 pc0_times[num_iter];
```
From the testbench host code, we configure the hardware performance counters (4 counters per AIE core, 2 counters per AIE local memory), and set `Start` and `Stop` event triggers for our counters to count the number of cycles between the two triggers. 
> Note that the event triggers for `Start` and `Stop` work much like a stopwatch where multiple start/stop pairs are cumulative and the next start signal restarts the counter at the value it stopped on. In addition, new `Start` events while the counter is counting is ignored and new `Stop` while the counter is paused likewise has no effect. This also means that if the counter is started and not stopped, the value you get when you read will be invalid.

In this example, we call `XAie_EvenPCEnable` to define an event based off two program counter values: 0 (program start) and 240 (program end). These two performance counter values are true for every AIE program but can change as the compiler evolves (accurate for Vitis 2022.2). These events are assigned to event 0 and event 1 for PC events.

```
XAie_EventPCEnable(&(_xaie->DevInst), XAie_TileLoc(1,4), 0, 0);
XAie_EventPCEnable(&(_xaie->DevInst), XAie_TileLoc(1,4), 1, 240);
```
Now, we configure the performance counter to have `Start` and `Stop` values based on the two defined performance counter events, being sure that we set this prior to the core being run.
```
EventMonitor pc0(_xaie, 1, 4, 1, XAIE_EVENT_PC_0_CORE, XAIE_EVENT_PC_1_CORE,
                    XAIE_EVENT_NONE_CORE, XAIE_CORE_MOD);
pc0.set();
```
This `EventMonitor` class is a wrapper to simplify the commands needed to set up the performance counter. We pass in our AIE config struct (_xaie), tile column (1), tile row (4), performance counter number (1), start event trigger, end event trigger, reset event trigger, and counter mode (core).
> Note that event triggers is a larger topic which we will touch upon in later tutorials. For now, we have the performance counter events for PC 0 and PC 1 as well as no event (XAIE_EVENT_NONE_CORE). We also have counter modes to indicate which category of counters we're using (core, memory and PL section of the AIE tile).

We call the `set` class function to record the counter start value. Now we enable our AIE tiles so the design can run. After that, we compute the performance counter difference and store the value in our array of times with:
```
pc0_times[0] = pc0.diff();
```
The `diff` class functions calculates the differnece in the counter from the last `set` or `diff` call. We can call it multiple times and store the results from mutliple runs to see if the values vary. We finish our program by reporting the number of cycles captured by our performance counters by calling:
```
computeStats(pc0_times, 1);
```
This is passed the array of times along with size of the array in order to report an average of the values.

10. Run make of the performance example to compile a design that will run on the board and report kernel cycles count performance to the terminal.
    ```
    make tutorial-1_perf.exe
    ```
    How many cycles is reported by the performance counter? <img src="../images/answer1.jpg" title="6" height=25>

This number includes program initalization and cleanup on top of the actual kernel code. As kernel code is often run in a loop or for multiple iterations, this initialization and cleanup code cost is amortized when the design is running in steady state. We can subtract the initialiation and cleanup cycles by building an empty design and counting its reported cycles (as seen [here](../../test/benchmarks/05_Core_Startup/)). For Vitis 2022.2, this baseline is 128 cycles, which means the kernel code was absorbed in the initialziation and cleanup code cycles.

We are now ready to start talking about how communication between AI Engine tiles is modeled in MLIR-AIE in [tutorial-3](../../tutorial-3).


