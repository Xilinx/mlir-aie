You will implement code that generates three variants of a SwiGLU operator and host test code in three subfolders:

 - `separate_xclbins`
 - `runlist`
 - `fused_transactions`

Each will implement a sequence of operations (kernels) run on the NPU device. Your job is to generate 
(a) a Makefile that compiles the necessary artifacts for each of the above variants (xclbins, combined xclbins and ELF)
(b) testing code (a test.cpp) that invokes that code a number of iterations and times those runs
(c) a script that runs all variants, collects the timing information, and stores results to a CSV
(d) a script that visualizes the above results in a plot and writes some statistics out to stdout comparing the approaches.

Write the Makefiles so they are somewhat reusable for similar-enough operators. At the top of the makefile, add variables for the kernel names, and output and input file names. You may assume future operators will have the same number of xclbins, so it's fine to hard code that (but make it easy to copy-paste and add more if needed).

Make the `test.cpp` similarly reusable: Hard-code some of the assumptions (number of kernels/operations in the sequence), but allow for some flexibility/make the code easy to edit if we need to add more kernels or more buffers.

Model your implementation after the code examples, simplifying and being more concise where possible. Don't add unnecessary comments.

# Code examples

## SwiGLU-specific example

- `swiglu_runlist_build_commands.sh` shows an example of how the SwiGLU kernels and xclbins can be compiled into a combined xclbin. An operator compiled like this can be used in a runlist. You will have to adapt this code to work with the current directory structure (kernels in kernels dir, Python files in operators dir) and adjust, based on the below examples, for the other non-runlist variants.

## Separate xclbin example

- `/test/npu-xrt/add_one_two/run.lit` - shows the compilation commands needed to create the xclbins
- `/test/npu-xrt/add_one_two/test.cpp` - host code for running  a single xclbin. You can skip the "get the kernel from the xclbin" verification part.

## Runlist example

- `/test/npu-xrt/add_one_two_runlist/run.lit` - compilation commands -- similar to the above for SwiGLU
- `/test/npu-xrt/add_one_two_runlist/test.cpp` shows some code on how to create host C++ code that invokes an example as compiled for the runlist. Note however that you can skip the "get the kernel from the xclbin" part -- that really doesn't do much, since it just checks whether a kernel of the given string name is in the xclbin. Instead, just use the given string names directly.

## Fused transactions example

This one is the most different; rater than an xclbin, there will be a single ELF file. To compile that ELF file, you will have to combine multiple MLIR files, give the devices in them different names, and create a new device called @main (name can be left off for main as it is the default) that uses aiex.configure instructions, does the proper buffer slicing and calls into the other devices runtimes. Don't manually combine the MLIR files; write a script that does this.

- `/test/npu-xrt/reconfigure/aie.mlir` -- required input MLIR structure (note different devices, aiex.configure, aiex.run and aiex.arg_slice)
- `/test/npu-xrt/reconfigure/test.cpp` -- how to call such a "full ELF" flow fused transaction example
- `/test/npu-xrt/reconfigure/Makefile` -- how to build for this example. The Makefile is the most up-to-date; don't look at `run.lit` here as it might be misleading.

For this approach, I'd also like you to just create one large in/out buffer that is passed in to the main runtime sequence. Then slice it into appropriate-sized subslices in both the `test.cpp` and the `aie.mlir`.

# Runtime Setup for SwiGLU (Decode)

The following are the concrete kernels to call from the `test.cpp` and what buffers need to be allocated.

Assume the following concrete parameters.
In the code, define these once as a constant, then use that constant.

 - embedding_dim = 2048
 - hidden_dim = 8192

Also define:
 - dtype: std::bfloat16_t

then use sizeof(dtype) to determine the byte size of buffers.

## Required buffers

The following are the buffers required by the kernels, with buffer names and shapes listed.
All data is in bfloat16, i.e. 2 bytes per element.
For buffer allocation, these should be allocated as a "flat" sequential row-major buffer (i.e. product of shape).

 - "input" (embedding_dim, )
 - "weights_1", (embedding_dim, hidden_dim, )
 - "weights_2", (embedding_dim, hidden_dim, )
 - "weights_3", (hidden_dim,  embedding_dim, )
 - "left", (hidden_dim, )
 - "left_swished", (hidden_dim, )
 - "right", (hidden_dim, )
 - "intermediate", (hidden_dim, )
 - "output", (embedding_dim, )

## Kernels

These correspond to individual xclbins if compiled separately, or to different kernels within one xclbin if chained together using `--xclbin-input` (in that case use the last xclbin of the chain which will contain all others).

 - "swiglu_gemv_1"
 - "swiglu_silu"
 - "swiglu_eltwise_mul"
 - "swiglu_gemv_2"

## Runlist (sequence of operations)

The operations need to be run in this sequence.
The format of the following is kernel_name, argument buffer 1, argument buffer 2, argument buffer 3, ...

 - "swiglu_gemv_1", "weights_1", "input", "left"
 - "swiglu_gemv_1", "weights_2", "input", "right"
 - "swiglu_silu", "left", "left_swished"
 - "swiglu_eltwise_mul", "left_swished", "right", "intermediate"
 - "swiglu_gemv_2", "weights_3", "intermediate", "output"
