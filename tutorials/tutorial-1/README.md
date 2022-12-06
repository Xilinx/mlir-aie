<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 1 - modules, tile, buffer, core, kernel</ins>
In the MLIR-based AI Engine representation, every physical component of the AI Engine array including connections are declared within the top level block called a `module`. All parameters and customizations of these components are then elaborated within the `module`. We generally write the MLIR code in a file with the .mlir file extension as it integrates well with the lit based auto-test of LLVM, such as those found in `test` sub-folder. An module declaration is shown below:

```
module @module_name {
    ... 
    AI Engine array components and connections 
    ...
}
```

1. Take a look at the source file [aie.mlir](aie.mlir) and identify the module name used in tutorial 1. <img src="../images/answer1.jpg" alt="tutorial_1" height=25>

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
>**Note:** index values start at 0, with row 0 belonging to the shim which is not a full regular row. The first regular row for first generation AI engines is row index 1. 
Tile declaration is mainly needed so other sub components can be associated to the tile by name. Some higher level logical components may also automatically declare tiles so they are enabled (e.g. logical flows require the tile to be enabled to support stream switch routing).

The type of tiles and orientation of its associated local memory is architecture dependent. In the case of the first generation of AI Engines, each tile has an associated local memory physically to its left or right depending on the row.
* odd row - memory on left
* even row - memory on right

<p><img src="../images/diagram3.jpg?raw=true" width="500"><p>

### <ins>Buffer</ins>

When delcaring a buffer, we pass in associated AIE tile and declare the buffer parameters. Those parameters are the depth and data type width (though the local memory itself is not physically organized in this way). 
> One important note about buffers is that they do not strictly map one buffer to one local memory. You can declare multiple buffers that are associated with a tile and they would by default be allocated sequentially in that tile's local memory.

### <ins>Core</ins>

The AIE core functionality is defined within the core body. More details on this is in tutorial 2.

We will be introducing more components and the ways these components are customized in subsequent tutorials. Additional syntax for these MLIR-based AI Engine components can be found in the github<area>.io docs [here](https://xilinx.github.io/mlir-aie/AIEDialect.html).

## <ins>Build tutorial 1 design</ins>
2. After building and installing `mlir-aie`, run make to compile the first design.
    ```
    > make
    ```
    This will run the kernel compilation tool (xchesscc) on the kernel code for a single AI Engine which will be covered in more detail in tutorial 2. It then run the python script which executes the generated `mlir-aie` tools for compiling our design from the `aie.mlir` source.
    
3. Take a look at `aie.mlir` to see how we mapped externally compiled AIE kernel objects. What name is the external function set to? <img src="../images/answer1.jpg" alt="extern_kernel" height=25>. Note that this name does not have to match the actual kernel name but is used in our mlir file to reference a particular defined function. The function arguments though, do have to match the external function for it to be succesfully integrated.
    > There's is no current error checking to ensure this mapping matches

4. The core is then linked to an object file where the function is defined. What is the name of the object file that the core in tile(1,4) is defined in? <img src="../images/answer1.jpg" alt="kernel.o" height=25> Matching kernel object files are necessary in order for successful elf integration at a later build stage. 

5. Take a look at the kernel source in `kernel.cc`. What value does the kernel set the value of index 3 of buffer `buf` to? <img src="../images/answer1.jpg" alt="14" height=25>

In addition to the generated core program (core_1_4.elf), to run our design on the board requires a host program which configures the AIE array and enables the AIE cores to run their individual programs. This host program can also serve as a testbench.

6. Take a look at `test.cpp`. There are a number of configuration function called to initialize and configure the AIE array. These functions are defined in the generated `acdc_project/aie_inc.cpp` file. Read through this testbench to see the explanations of what each helper function does and how it check the results after confiuring and enabling the AIE cores. 

7. **PLACEHOLDER** Run simuation

8. Copy the generated executables (tutorial-1.exe, core_1_4.elf) to the vck190 board and run the test bench executable (tutorial-1.exe) to see that the compiled program works.

## 

