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

The MLIR-AIE dialect builds a number of command utilities for compiling and transforming operations written in the MLIR-AIE dialect into other intermediate representations (IRs) as well as generating AIE program elfs and host executables to be run on the board. The two main output utilities that building the MLIR-AIE project gives you is `aie-translate` adn `aie-opt` which are used to transform MLIR-AIE dialect into other IRs. These utilities are then used by the convenience python utility `aiecc.py` to compile operations written in MLIR-AIE dialect into elf/host executables.

## <ins>aiecc.py</ins>

The basic way that we use `aiecc.py` to compile our MLIR-AIE written code (e.g. aie.mlir) into elf/host executable (core_*.elf, test.exe) is the following:
```
aiecc.py -j4 --sysroot=<platform sysroot> -host-target=aarch64-linux-gnu aie.mlir -I<runtime_lib> <runtime lib>/test_library.cpp ./test.cpp -o test.exe
```
This command compiles an MLIR-AIE dialect source file (e.g. `aie.mlir`) into AI Engine program elfs and generates the host API `aie.mlir.prj/aie_inc.cpp` which can be used to configure the AI Engine design. In addition, if provided with a host source file (e.g. `test.cpp`), `aiecc.py` will compile it to build a host executable (e.g. `tutorial-1.exe`). The AIE tile elfs are generated automatically for each AIE tile that needs to be programmed. Additionally, we usually pass a reference to`<runtime lib>/test_library.cpp` as it contains commonly used test functions.

You can see what arguments are supported with `aiecc.py` by calling it with the -h argument. Some useful arguments are listed below:

| Optional arguments | Description |
|--------------------|-------------|
|  --sysroot sysroot |    sysroot for cross-compilation |
|  -v                |    Trace commands as they are executed |
|  --vectorize      |     Enable MLIR vectorization |
|  --xbridge        |     Link using xbridge (default) |
|  --xchesscc       |     Compile using xchesscc (default) |
|  --compile        |     Enable compiling of AIE code (default) |
|  --no-compile      |    Disable compiling of AIE code |
|  --host-target HOST_TARGET | Target architecture of the host program (e.g. vck190 uses aarch64-linux-gnu) |
|  --compile-host   |     Enable compiling of the host program (default) |
|  --no-compile-host|     Disable compiling of the host program |
|  --link           |     Enable linking of AIE code (default) |
|  --no-link        |     Disable linking of AIE code |
|  -j NTHREADS      |     Compile with max n-threads in the machine (default is 1). An argument of zero corresponds to the maximum number of threads on the machine. |
|  --profile        |     Profile commands to find the most expensive executions. |
|  --unified        |     Compile all cores together in a single process (default) |
|  --no-unified     |     Compile cores independently in separate processes |
|  -n               |     Disable actually executing any commands. |

## <ins>aie-opt</ins>

This is the primary utility does transforms/ optimizes source code from one representation to another within the MLIR-AIE defined dialect. This is primarily controlled via command line options and adding multiple options allows the utility to perform multiple transformations/ optimizations. A full description of the options can be found [here](https://xilinx.github.io/mlir-aie/AIEPasses.html), but some example options transforms from a logical description to a more physical one such as in between flow and switchboxes in [tutorial-4](../tutorial-4)

## <ins>aie-translate</ins>

This utility is more geared toward translating a description into another format altogether. For example, generating accessory files like .bcf and .ldscript would be done with this utility. Again, the full description of options can be found [here](https://xilinx.github.io/mlir-aie/AIEPasses.html). 


## <ins>Walking through aiecc.py</ins>

Rather than walk through the source code for `aiecc.py`, we will describe some of the main calls that `aiecc.py` makes to `aie-translate` and `aie-opt` and show the arguments used to provide some idea about what they do.

The main flow of `aiecc.py` is:
1. First set of optimizations
2. Translate file to count AI Engine cores in design
3. Second set of optimization
4. Translate in LLVM-IR
5. Compile invididual cores (e.g. xchesscc_wrapper)
6. Process ARM cross-compilation of host code
7. for loop over cores

    8. 1st set of core optimizations
    9. 2nd set of core optimizations
    10. Translate to generate .bcf or .ldscript 
    11. Compile core (e.g. xchesscc_wrapper)

### <ins>1. First set of optimizations</ins>
```
aie-opt 
--lower-affine
--aie-canonicalize-device
--aie-assign-lock-ids
--aie-register-objectFifos
--aie-objectFifo-stateful-transform
--aie-lower-broadcast-packet
--aie-lower-multicast
--aie-assign-buffer-addresses
--convert-scf-to-cf
aie.mlir -o input_with_addresses.mlir
```
### <ins>2. Translate file to count AIE Engine cores in design</ins>
```
aie-translate --aie-generate-corelist input_with_addresses.mlir
```

### <ins>3. Second set of optimization</ins>
```
aie-opt 
--aie-localize-locks
--aie-standard-lowering
--aie-normalize-address-spaces
--canonicalize
--cse
--convert-vector-to-llvm
--expand-strided-metadata
--lower-affine
--convert-arith-to-llvm
--convert-memref-to-llvm
--convert-func-to-llvm=use-bare-ptr-memref-call-conv
--convert-cf-to-llvm
--canonicalize --cse
input_with_addresses.mlir -o input_opt_with_addresses.mlir
```

### <ins>4. Translate into LLVM-IR
```
aie-translate 
--opaque-pointers=0 
--mlir-to-llvmir 
input_opt_with_addreses.mlir -o input.ll
```
### <ins>5. Compile individual cores (e.g. xchesscc_wrapper)</ins>
```
xchesscc_wrapper -c -d -f +P 4 file_llvmir_hacked -o input.o
```

### <ins>6. Process ARM cross-compilation of host code</ins>
This section does cross-compilation which includes calling clang with a specific target. However, `aie-opt` and `aie-translate` is called here to generate the `aie.mlir.prj/aie_inc.cpp` file.
```
aie-opt
--aie-create-pathfinder-flows
--aie-lower-broadcast-packet
--aie-lower-multicast
input_with_addresses.mlir -o input_physical.mlir
aie-translate --aie-generate-xaie --xaie-target=v2 input_physical.mlir -o aie_inc.cpp
```

### <ins>7. 1st set of core optimizations</ins>
```
aie-opt 
--aie-localize-locks 
--aie-standard-lowering=tilecol=COL tilerow=ROW % core[0:2] 
input_with_addresses.mlir -o  -o core_*.mlir
```

### <ins>8. 2nd set of core optimizations</ins>
```
aie-opt
-aie-normalize-address-spaces
--canonicalize
--cse
--convert-vector-to-llvm
--expand-strided-metadata
--lower-affine
--convert-arith-to-llvm
--convert-memref-to-llvm
--convert-func-to-llvm=use-bare-ptr-memref-call-conv
--convert-cf-to-llvm
--canonicalize
--cse 
core*.mlir -o opt.mlir
```

### <ins>9. Translate to generate .bcf or .ldscript</ins>
```
aie-translate input_with_addresses.mlir --aie-generate-bcf --tilecol=COL --tilerow=ROW -o core*.bcf
or 
aie-translate input_with_addresses.mlir --aie-generate-ldscript --tilecol=COL --tilerow=ROW -o core*.ld.script
```


### <ins>10. Compile core (e.g. xchesscc_wrapper)</ins>
This section makes calls to `xchesscc_wrapper` to compile the final core*.elf.
```
xchesscc_wrapper -d -f core*.o link_with_obj +l core*bcf -o core*elf
```

    
