<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 1 - modules, tile, buffer, core</ins>
In the MLIR-based AI Engine representation, every physical component of the AI Engine array including connections are declared within the top level block called a `module`. All parameters and customizations of these components are then elaborated within the `module`. We generally write the MLIR code in a file with the .mlir file extension as it integrates well with the lit based auto-test of LLVM, such as those found in `test` sub-folder. A module declaration is shown below:

```
module @module_name {
    ... 
    AI Engine array components and connections 
    ...
}
```

## <ins>Tile Components</ins>
AI Engine tiles are the basic building blocks of AIE designs and can be declared as `AIE.tile(col,row)`. Examples incude:
```
%tile13 = AIE.tile(1,3)
%tile23 = AIE.tile(2,3)
%tile33 = AIE.tile(3,3)
```
The two major components of an AI Engine tile is 

* VLIW processor core declared as `AIE.core(tileName) { body }`
* Local memory buffer declared as `AIE.buffer(tileName) : memref<depthxdata_type> { body }`. 

Examples declarations include:
```
AIE.core(%tile13) {
    ... 
    core body 
    ...
}

%buff0 = AIE.buffer(%tile13) : memref<256xi32>
%buff1 = AIE.buffer(%tile13) : memref<256xi32>
```
The association between these declarations and the physical AI Engine tile components can be seen here. For more details on mlir-aie dialect syntax, you can refer to the onine reference document [here](https://xilinx.github.io/mlir-aie/AIEDialect.html).
<img src="../images/diagram1.jpg?raw=true" width="800">

### <ins>Tile</ins>

For the tile, we simply declare its coordinates by column and row 
>**Note:** index values start at 0, with row 0 belonging to the shim which is not a regular row. The first regular row for first generation AI engines is row index 1. 
Tile declaration is mainly needed so other sub components can be associated to the tile by name. Some higher level logical components may also automatically declare tiles so they are enabled (e.g. logical flows require the intermediate tiles along the path to be enabled to support stream switch routing).

The type of tiles and orientation of its associated local memory is architecture dependent. In the case of the first generation of AI Engines, each tile has an associated local memory physically to its left or right depending on the row.
* odd row - memory on left
* even row - memory on right

<p><img src="../images/diagram3.jpg?raw=true" width="500"><p>

### <ins>Buffer</ins>

When delcaring a buffer, we pass in associated AIE tile and declare the buffer parameters. Those parameters are the depth and data type width (though the local memory itself is not physically organized in this way). 
> One important note about buffers is that one buffer is not strictly mapped to the entire local memory. You can declare multiple buffers that are associated with the local memory of a tile and they would, by default, be allocated sequentially in that tile's local memory.

### <ins>Core</ins>

The AIE core functionality is defined within the core body. This functionality is a combination of AIE dialect specific operations as well as other general dialects that are supported by the MLIR compiler. This includes a large set of dialects such as [arith](https://mlir.llvm.org/docs/Dialects/ArithOps/) and [memref](https://mlir.llvm.org/docs/Dialects/MemRef/) but can also include many others. Custom functions that are not inherently supported on AI Engines can be translated into scalara operations that are (e.g. arctan). Keep in mind that MLIR is not a programming language but a intermediate representation so the syntax of doing simple opertions may seem cumbersome at first glance but is designed to capture a robust set of operations.

In addition to the integrated core functionality defintions, `mlir-aie` also supports linking with externally compiled kernel code which we will go into more detials in tutorial 2. This process allows custom kernels to be included directly in `mlir-aie` defined designs.

We will be introducing more components and the ways these components are customized in subsequent tutorials. Additional syntax for these MLIR-based AI Engine components can be found in the github<area>.io docs [here](https://xilinx.github.io/mlir-aie/AIEDialect.html).

## <ins>Tutorial 1 Lab</ins>
1. Take a close look at the source file [aie.mlir](aie.mlir) and read through the code comments which walks through the syntax of this simple design. What is the variable name of the module? <img src="../images/answer1.jpg" title="tutorial_1" height=25>

2. After building and installing `mlir-aie`, run make to compile the first tutorial.
    ```
    > make
    ```
Under the hood, `make` calls `aiecc.py` which itself calls a number of utilities that are built as part of the `mlir-aie` project (`aie-translate`, `aie-opt`). The MLIR operations inside the core are then converted to an LLVM representation which the AMD internal compiler (currently xchesscc) takes and builds the executable that will run on each individaul AIE tile. 
   
3. In [aie.mlir](aie.mlir), what is the variable name for tile(1,4)? <img src="../images/answer1.jpg" title="%tile14" height=25> 

    What about the variable name and size of the buffer that is associated with the local memory of tile(1,4)? <img src="../images/answer1.jpg" title="%buf, 256 x int32" height=25>

In first generation AI Engines, each tile has 32 kB of local data memory assigned to it. In addition, it can access the local memory in the other 3 cardinal directions giving a total accessible memory of 128 kB.

4. What percentage of the local memory (32 kB) does this design currently use?
<img src="../images/answer1.jpg" title="3%, 256 / 8192 int32" height=25>

5. Change the size of the buffer to the size of our local memory (8192 x i32) and run `make` again. What do you expect to happen and what happens instead? <img src="../images/answer1.jpg" title="Allocated buffers exceed local memory." height=25>

While we have a separate 16 kB of program memory which stores the AIE program code, the 32 kB of data memory is also used for the program stack. By default, the tool reserves 4096 bytes for the stack so all buffers are then allocated immediately after that. 

6. Declare an horizontally adjacent tile (pay attention to which row we're in) so that tile (1,4) can access the neighbor tile's local memory. Declare a buffer in this tile that uses the entire local memory (8192 x i32) and replace the reference %buf in line 34 with the new buffer and run `make ` again. What do you expect to happen and what happens instead? <img src="../images/answer1.jpg" title="We expect to be able to compile successfully since we can use our neighbor's full local memory but the same error shows up. Right now, all tiles that are enabeld have a stack space of 4096 bytes reserved, even if that tile is not running any program code. This is an area for future improvement." height=25>

To create complete AI Engine designs that can run on a Versal device, we also need code that is used to configure and program the AIE array. `aiecc.py` is used to translate the physical MLIR description of our design into elf files that run on each tile (e.g. core_1_4.elf). It also generates configuration and initialization functions under `acdc_project/aie_inc.cpp`. These functions can be invoked as part of a host code testbench [test.cpp](test.cpp) which can be run on a target platform such as the the vck190 development board. While running our `mlir-aie` tools on an x86 machine, we can cross-compile the final executable (`tutorial-1.exe`) to run on the ARM processor on the Versal device.

7. Take a look at  [test.cpp](test.cpp). There are a number of configuration functions for initializing and configuring the AIE array. These functions are defined in the test library [test_library.h](../../runtime_lib/test_library.h) and the generated `acdc_project/aie_inc.cpp` file. Read through this testbench to see the explanations of what each helper function does and how we can check the results after configuring and enabling the AIE cores. What is the expected value at buf[3] that we're checking for after our design is run? <img src="../images/answer1.jpg" title="14" height=25>

8. **PLACEHOLDER** Run simuation

9. Copy the generated executables (tutorial-1.exe, core_1_4.elf) to the vck190 board and run the test bench executable to see that the compiled program works on the board.
    ```
    sudo ./tutorial-1.exe
    ```
## <ins>Advanced Topic - Performance Measurement in HW</ins>
Now that we've compiled, simulated and run our design. We can take a step back to leverage some of the AI Engine's built in performance measurement hardware (timers and event traces) to capture actual on-board performance. 

We can first declare some host code variable to hold timer values.
```
u32 pc0_times[num_iter];
```
From the testbench host code, we configure the hardware program counters (2 counters per AIE core, 4 counters per AIE local memory), and set `Start` and `Stop` event triggers for our counters to count the number of cycles between the two triggers. 
> Note that the event triggers for `Start` and `Stop` work much like a stopwatch where multiple start/stop pairs are cumulative and the next start signal restarts the counter at the value it stopped on. In addition, new `Start` events while the counter is counting is ignored and new `Stop` while the counter is paused likewise has no effect. This also means that if the counter is started and not stopped, the value you get when you read will be invalid.

In this example, we call `XAie_EvenPCEnable` to define an event based off two program counter values: 0 (program start) and 240 (program end). These two program counter values are true for every AIE program but can change as the compiler evolves (accurate for Vitis 2022.2). These events are assigned to event 0 and event 1 for PC events.

```
XAie_EventPCEnable(&(_xaie->DevInst), XAie_TileLoc(1,4), 0, 0);
XAie_EventPCEnable(&(_xaie->DevInst), XAie_TileLoc(1,4), 1, 240);
```
Now, we configure the program counter to have `Start` and `Stop` values based on the two defined program counter events, being sure that we set this prior to the core being run.
```
EventMonitor pc0(_xaie, 1, 4, 1, XAIE_EVENT_PC_0_CORE, XAIE_EVENT_PC_1_CORE,
                    XAIE_EVENT_NONE_CORE, XAIE_CORE_MOD);
pc0.set();
```
This `EventMonitor` class is a wrapper to simplify the commands needed to set up the program counter. We pass in our AIE config struct (_xaie), tile column (1), tile row (4), program counter number (1), start event trigger, end event trigger, reset event trigger, and counter mode (core).
> Note that event triggers is a larger topic which we will touch upon in later tutorials. For now, we have the program counter events for PC 0 and PC 1 as well as no event (XAIE_EVENT_NONE_CORE). We also have counter modes to indicate which category of counters we're using (core, memory and PL section of the AIE tile).

We call the `set` class function to record the counter start value. Now we enable our AIE tiles so the design can run. After that, we compute the program counter difference and store the value in our array of times with:
```
pc0_times[0] = pc0.diff();
```
The `diff` class functions calculates the differnece in the counter from the last `set` or `diff` call. We can call it multiple times and store the results from mutliple runs to see if the values vary. We finish our program by reporting the number of cycles captured by our program counters by calling:
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

