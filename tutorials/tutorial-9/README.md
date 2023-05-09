<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->
# <ins>Tutorial 9 - Single kernel compilation and simulation</ins>

MLIR gives us the ability to leverage different dialects such as [arith](https://mlir.llvm.org/docs/Dialects/ArithOps/) and [memref](https://mlir.llvm.org/docs/Dialects/MemRef/) when defining AIE core functionality. The ability to lower from these dialects and many others into efficient AI Engine code is an active area of research and development. However, when working with existing optimized AIE kernel code written in C/C++ (or wanting to write your own), we can reference precompiled object code in our AIE core operation definition through the use [func](https://mlir.llvm.org/docs/Dialects/Func/) dialect.

## <ins>MLIR external functions</ins>

Specifically, to support external functions, we use the operators `func.func` and `func.call` as follows:
```
func.func private @extern_kernel(%b: memref<256xi32>) -> ()

%core14 = AIE.core(%tile14) {
    func.call @extern_kernel(%buf) : (memref<256xi32>) -> ()
    AIE.end
} { link_with="kernel.o"}
```
In this MLIR code snippet, we see that we first call `func.func` to declare a private function whose function signature matches that of the AIE C/C++ function. The function name after the @ (e.g. `@external_kernel`) should match the C function name and the number of arguments should match the number of C function arguments.  C++ name mangling is not supported.  Argument types are converted according to the MLIR ['bare pointer' calling convention](https://mlir.llvm.org/docs/TargetLLVMIR/#bare-pointer-calling-convention-for-ranked-memref) (see below). 

| MLIR type   | C type      |
| ----------- | ----------- |
| i32         | int32_t     |
| f32         | float       |
| Memref      | C pointer   |
| index       | int64_t     |

Then, within the `AIE.core` operator, we use `func.call` to call the previously defined function from within our core, being sure to pass the appropriate function arguments. In this case, we pass in the the `AIE.buffer` `%buf`. 

The final step is to tell our tools where to look for the object code that the function whose name we defined in `func.func`/ `func.call`. Using the additional operator definition `link_with="kernel.o"`, we point to the file `kernel.o` in the current directory and link it in to create the final kernel object file.
> Note that this allows us to call the function multiple times within the `AIE.core` or even separate functions in the same `AIE.core` if they are both defined within the single linked object file.

## <ins>Kernel object file generation</ins>

Now that we know how to link in externally defined functions from precompiled object files, it would be nice to be able to compile those object files quickly as well as test them.  You may be familiar with using the Vitis GUI to, but we can also directly call the underlying compilation tools used by Vitis.

To compile the C/C++ source into object code, we use the `xchesscc` command line tool as follows:
```
xchesscc -p me -P <vitis install>/<release>/aietools/data/<aie-version>/lib -c kernel.cc
```
Within `kernel.cc`, the function must be defined as extern "C" with:
```
extern "C" {
    <function definition>
}
```
Place the `kernel.o` in the same directory as your MLIR source and you should now be able to run the build tools to generate the new aggregated object file.

## <ins>Single kernel compilation and simulation</ins>

The last step is working with C/C++ source to compile and simulate the kernel. This main way to do this is once again with the Vitis tools, specifically aiecompiler and aiesimulator which requires an adf graph definition, even for a single kernel function. However, we can again directly call the compilation and simulation tools that Vitis uses. 

To be able to compile and test, we need a testbench wrapper around our kernel, similar to the testbench and platform defined for an `adf` design. In our example external function, `external_kernel`, we define the following files:
```
kernel.h - kernel header file
test.cc  - testbench that calls our kernel (similar to test.cpp)
test.prx - XML project file
```
Once these file are defined, we call the `xchessmk` command line tool as follows:
```
xchessmk -P <vitis install>/<release>/aietools/data/<aie-version>/lib test.prx
```
This compiles our testbench and kernel into a default `work` directory so that it is ready to simulate, which we do so by calling the `xca_udm_dbg` command line tool as follows:
```
xca_udm_dbg -P <vitis install>/<release>/aietools/data/<aie-version>/lib -t sim.tcl
```
This simulator executes a number of Tcl commands which we can group into Tcl batch file called  `sim.tcl`. The cycle accurate simulator will run the commands in this tcl file to completion and outputs any testbench results. This allows us to iteratively compile and test our design multiple times to get the right behavior as well as profile code performance.
> Note that in `sim.tcl`, we call the command `iss profile save test.prf` which runs the profiler in our simulator and generates the profile summary file `test.prf`. We will look at this in more detail in the lab.

Now we have all the pieces we need to compile and simulate single kernels from the command line and then compile the kernel core into an object file to be integrated into our MLIR description and expanded into full kernel object code.

## <ins>Tutorial 9 Lab</ins>

1. We will work backwards in this lab to first compile and simulate our single kernel design (which is the duplicate of the simple design used in tutorial-1). Take a look at the file under the `external_kernel` directory, specifically the files [kernel.h](external_kernel/kernel.h), [test.cc](external_kernel/test.cc), and [test.prx](external_kernel/test.prx) to familiarize yourself with these file contents. When customizing this for your own kernel function, you will only need to modify `kernel.h` (to match your function signature) and `test.cc` (to customize the function call and testbench for your own function).

2. Go into the `external_kernel` directory and compile and simulate as follows:
    ```
    > make build
    > make sim
    ```
    How many cycles did the simulation take? <img src="../images/answer1.jpg" title="214 cycles" height=25>

3. Take a look at the simulation script [sim.tcl](external_kernel/sim.tcl). The command `iss profile save test.prf` invokes the profiler and saves profile information into the `test.prf` file. Open `test.prf`. What is the `Total cycle count` of our design? <img src="../images/answer1.jpg" title="Also 214 cycles" height=25>

4. The profile information also breaks down the cycle count of the design and more importantly, the function we implemented. What is the cycle count of the function `extern_kernel`? <img src="../images/answer1.jpg" title="6 cycles" height=25>
There is also the microcode of the function under `Function detail: extern_kernel`. Understanding the microcode can be helpful in maximally optimizing your AI Engine kernel but that is beyond the scope of this tutorial.

5. There is another design under the `matmul_kernel` directory. Build and simulate this design. Based on the profile results, what is the cycle count of main() and that of the matmul function (also called `extern_kernel`)? <img src="../images/answer1.jpg" title="testbench cycle count = 5128 cyles. extern_kernel = 4978 cycles" height=25>

6. Take a look at [aie.mlir](aie.mlir) to see how we used the [func](https://mlir.llvm.org/docs/Dialects/Func/) dialect to map externally compiled AIE kernel objects. Run `make` to build our MLIR design.

7. Verify functionality by running simulation
    ```
    make -C aie.mlir.prj/sim
    ```

8. Copy the design files (tutorial-2.exe, core_1_4.elf) to the board and run the design to check that the design runs successfully on hardware.

### <ins>Challenge Exercise</ins>
9. Change the design to use the kernel.o created by matmul_kernel. Note that the arguments are very different so you will need to use `mlir_aie_write_buffer_<bufname>` to initialize the local data memory and then `mlir_aie_read_buffer_<bufname>` to check the kernel results to verify correct functionality that we check in the [mlir_kernel/test.cc](mlir_kernel/test.cc). <img src="../images/answer1.jpg" title="See example host code under <answers> sub-directory. This can be built by running > make tutorial-9_matmul_perf.exe" height=25>
