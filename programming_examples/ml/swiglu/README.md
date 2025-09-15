# SwiGLU

This example demonstrates how we can chaing multiple designs together to 
implement a computation graph. The computation graph that we will show here is
SwiGLU (Swished Gated Linear Unit), an activation function used in some LLMs:

```
     (W1)  (input)  (W2)
       |    /   \    |
       |   /     \   |
       |  /       \  |
      GEMM       GEMM
      (=left)    (=right)
        |           |
        |           |
        |           /
      SiLU         /
(=left_swished)   /    
         \       /
          \     /
           \   /
        eltwise mul.
         (=result)
```

The designs invoked here are GEMM (general matrix-matrix multiplication),
SiLU (Sigmoid Linear Unit) and eltwise mul (elementwise multiplication).
The inputs intermediate results of these operations (the way they are named in
the code) are denoted in parantheses.

## Compiling multiple designs into one xclbin

In order to use the runlist, an XRT abstraction for more efficient invocation
of multiple designs, all designs must be compiled into the same NPU binary
(xclbin). With the `aiecc.py` compiler, you can achieve this using the 
`--xclbin-input` flag (to provide an input xclbin to add the current design to)
along with the `--xclbin-kernel-name`, `--xclbin-kernel-id` and 
`--xclbin-instance-name` flags, which are used to give each design in the
xclbin unique identifiers.

You can see how this is invoked in our Makefile target for the 
`combined.xclbin`:

```
	cd ${@D} && aiecc.py -v --no-xchesscc --no-xbridge --no-aiesim --peano ${PEANO_INSTALL_DIR} \
		--aie-generate-xclbin --xclbin-name=${DESIGN_2}.xclbin  \
		--aie-generate-npu-insts --npu-insts-name=${DESIGN_2}.bin \
		--xclbin-kernel-name=${DESIGN_2} --xclbin-kernel-id=0x903 --xclbin-instance-name=${DESIGN_2}_inst \
		--xclbin-input=${DESIGN_1}.xclbin \
		${DESIGN_2}.mlir
```

Note that we currently do not have a way to combine already-compiled xclbins,
though this would technically be feasible. Instead, you have to "chain" 
your compilations together in sequence using `--xclbin-input`, as shown above.

## Using the `InvocationPlan`

Once compiled, you could directly invoke your designs using the XRT library.
However, since chaining multiple designs together and piping outputs of one
design into the inputs of another requires a lot of boilerplate code at the 
XRT level, we abstracted this common use case into a small runtime API
called an `InvocationPlan`.

To set up an `InvocationPlan`, you must define (1) which designs you will want 
to use, along with the runtime sequence for each kernel, (2) what buffers 
these designs require (dimensions, name of the buffer, input/output direction,
optional reference data for verification) and (3) what the sequence of 
operations in terms of the defined designs and buffers should be.

Here is the relevant code from the `test.cpp` for our SwiGLU example:

First, we define the designs we want to use. Thse are all part of the same
xclbin, but each can have a different instruction sequence for the dedicated
command processor. The format is: 
`{<arbitrary name of your chosing>, <path to insts.bin>, <kernel name>}`.
It is critical that the kernel name (last field) matches the kernel name you
passed to aiecc.py with the `--xclbin-kernel-name` flag during compilation.
The first field (name) can be chosen freely and is what you will use later in 
the run list to refer to this kernel.
```
  std::vector<KernelInfo> kernels = {
    {"matmul",      vm["insts-matmul"].as<std::string>(),      "matmul"},
    {"silu",        vm["insts-silu"].as<std::string>(),        "silu"},
    {"eltwise_mul", vm["insts-eltwise-mul"].as<std::string>(), "eltwise_mul"}
  };
```

Then, we define each buffer, giving it an arbitrary name, passing the size of 
the buffer, the direction (input, output or inout) and, optionally, passing a
pointer to a reference buffer. If the buffer is an input, this reference
buffer will be fed to the design when it is launched; if it is an output, the 
verifier will check if the output from the NPU matched the reference buffer.
Note that the input/output designation is from a CPU perspective; if a buffer
is passed from one NPU design to the next, but does not need to be passed in
from the CPU, designate it as output.
```
  std::vector<KernelBufferInfo> buffers = {
    {"inp",          dim*seq, KernelBufferInfo::Direction::IN,     ptr_ref_inp},
    {"W1",           dim*dim, KernelBufferInfo::Direction::IN,     ptr_ref_W1},
    {"W2",           dim*dim, KernelBufferInfo::Direction::IN,     ptr_ref_W2},
    {"left",         dim*seq, KernelBufferInfo::Direction::OUT,    ptr_ref_left},
    {"right",        dim*seq, KernelBufferInfo::Direction::OUT,    ptr_ref_right},
    {"left_swished", dim*seq, KernelBufferInfo::Direction::OUT,    ptr_ref_left_swished},
    {"result",       dim*seq, KernelBufferInfo::Direction::OUT,    ptr_ref_result}
  };
```

Lastly, we define the runlist, which is of the format 
`{<kernel name>, {<arg buffer 1>, <arg buffer 2>, ...}}`:
```
  std::vector<KernelInvocationInfo> runlist = {
    {"matmul",      {"W1",           "inp",          "left"  }},
    {"matmul",      {"W2",           "inp",          "right" }},
    {"silu",        {"left",         "left_swished"          }},
    {"eltwise_mul", {"left_swished", "right",       "result"}}
  };
```

This fully defines our computation sequence. To invoke it we turn the 
`InvocationPlanInfo` into an `InvocationPlan`, which allocates all needed 
buffers and performs XRT setup, then call it:
```
  InvocationPlanInfo plan_info = {
      .xclbin = xclbin_path,
      .kernels = kernels,
      .buffers = buffers,
      .runlist = runlist
  };

  InvocationPlan plan = InvocationPlan::fromInfo(plan_info, device, xclbin, context);
  auto [success, t_elapsed] = plan.invoke();
```