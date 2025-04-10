<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Section 4b - Trace</ins>

* [Section 4 - Performance Measurement & Vector Programming](../../section-4)
    * [Section 4a - Timers](../section-4a)
    * Section 4b - Trace
        * [Trace Details - Placed and Unplaced Designs](./README-placed.md)
    * [Section 4c - Kernel Vectorization and Optimization](../section-4c)

-----

In the previous [section-4a](../section-4a), we looked at how timers can be used to get an overview of application performance. However, for kernel programmers that want to optimize the AIE hardware to its fullest potential, being able to see how efficiently the AIE cores and data movers are running is important. As such, the AIEs are equipped with tracing hardware that provides a cycle accurate view of hardware events. More detailed specification of the AIE2 trace unit can be found at in [AM020](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Trace).

Enabling trace support can be done with the following steps:

## <u>Steps to Enable Trace Support</u>
1. [Enable and configure trace](#1-enable-and-configure-aie-trace-units)
1. [Configure host code to read trace data and write it to a text file](#2-configure-host-code-to-read-trace-data-and-write-it-to-a-text-file)
1. [Parse text file to generate a waveform json file](#3-parse-text-file-to-generate-a-waveform-json-file)
1. [Open json file in a visualization tool like Perfetto](#4-open-json-file-in-a-visualization-tool-like-perfetto)
* [Additional Debug Hints](#additional-debug-hints)


## <u>1. Enable and configure AIE trace</u>

Enabling tracing means configuring the trace units for a given tile and then routing the generated event packets through the stream switches to the shim DMA where we can write them to a buffer in DDR for post-runtime processing. For our high-level IRON descriptions, we abstract these steps into a single runtime function `enable_trace` within the larger runtime sequence as shown below:
```python
    rt = Runtime()
    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (a_in, f_in, c_out):
        rt.enable_trace(trace_size, workers=[my_worker])
        ...
```
In this example, we pass as an argument the trace size and an array of workers that we want to trace since workers are associated with tiles. This allows us to trace all core tiles in our design. Tracing mem tiles and shim tiles at the moment requires us to work with explicitly placed designs using closer-to-metal IRON python, which is described in more detail in [README-placed](./README-placed.md).

An alternative to specifying the array of workers to trace would be to instead add a trace parameter to the worker declaration directly as shown below:
```python
    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_factor.cons(), of_out.prod(), scale],
        trace=1,
    )
    ...
    rt = Runtime()
    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (a_in, f_in, c_out):
        rt.enable_trace(trace_size)
        ...
```
Here, we addd `trace=1` to indicate that worker should be traced. And we can omit the `workers` argument from the `enable_trace` call in the runtime sequence.

>**NOTE**: The `workers` argument in the runtime sequence `enable_trace` always takes precendence over the `trace=1` argument of the woker. So if you define both, we will go with the definition of the `enable_trace` argument.

Configuring the trace unit in each core tile and routing the trace packets to a valid shim tile is then done automatically. In addition to the trace size and workers, other arguments currently supported by the `enable_trace` function are:
* *trace_offset* - byte offset for trace data in DDR memory
* *ddr_id* - XRT buffer we want to write to. See [below](#2-configure-host-code-to-read-trace-data-and-write-it-to-a-text-file) for more details on XRT buffers.

There are some assumptions and limitations to this automated process at the moment which will be elaborated more [README-placed](./README-placed.md). 

## <u>2. Configure host code to read trace data and write it to a text file</u>

Once the trace units are configured and routed, we want the host code to read the trace data from DDR and write it out to a text file for post-run processing. To give a better sense of how this comes together, this section provides an example design that is again a simplifed version of the [Vector Scalar Multiply example](../../../programming_examples/basic/vector_scalar_mul/).

### <u>AIE structural design code ([aie2.py](./aie2.py))</u>
In order to write the DDR data to a text file, we need to know where in DDR the trace data is stored and then read from that location. This starts inside the [aie2.py](./aie2.py) file where the `enable_trace` function under the hood expands to calls to configure the trace units and program the shimDMA to write to one of XRT inout buffers. This requires a more in-depth understanding about the *XRT buffer objects* described in [section 3](../../section-3). There we had described that our XRT supports up to 5 inout buffer objects. Common patterns include 1 input/ 1 output and 2 input/ 1 output. These patterns then map in the following way where the *group_id* is listed next to each XRT buffer object, `inoutN (group_id)`.

| inout0 (3) | inout1 (4) | 
|--------|--------|
| input A  | output C | 

| inout0 (3) | inout1 (4) | inout2 (5) |
|--------|--------|--------|
| input A  | input B | output C  |

To support trace, we will configure a shim tile to move the trace packet data to DDR through one of these XRT buffer objects. For simplicity, we choose `inout4 (7)` such that the new trace enabled mapping is:

| inout0 (3)| inout1 (4) | inout2 (5) | inout3 (6)| inout4 (7)|
|--------|--------|--------|--------|--------|
| input A  | output C | unused | unused | trace  |

| inout0 (3)| inout1 (4)| inout2 (5)| inout3 (6)| inout4 (7)|
|--------|--------|--------|--------|--------|
| input A  | input B | output C | unused | trace  |

In some designs, we have also used a pattern where we share an XRT buffer object where the trace data is written to same buffer object as the output (but with an offset). This is helpful if we do not have a spare buffer object dedicated to trace, but requires precise declaration of offset size.

| inout0 (3)| inout1 (4)| inout2 (5)| 
|--------|--------|--------|
| input A  | input B | (output C + trace) |

When using the convenience python wrappers, where 1 input/ 1 output and 2 inputs/ 1 output patterns are supported, we choose to map the trace data to `inout4(7)`. More details are described in [python/utils/xrt.py](../../../python/utils/xrt.py) including the option to map trace to a different XRT buffer object with a specified offset. 

Once [aie2.py](./aie2.py) is configured to output trace data to the 5th inout buffer, we turn our attention to the host code to read the DDR data and write it to a file.

> **NOTE** In our example design ([aie2.py](./aie2.py)), we provide a [Makefile](./Makefile) target `run` for standard build and `trace` for trace-enabled build. The trace-enabled build passes the trace buffer size as an argument to [aie2.py](./aie2.py) which is used under the hood to conditionally enable tracing as long as `trace_size` is > 0. This is also true for the [Vector Scalar Multiply example](../../../programming_examples/basic/vector_scalar_mul).

### <u>(2a) C/C++ Host code ([test.cpp](./test.cpp), [../../../runtime_lib/test_lib/xrt_test_wrapper_.h](../../../runtime_lib/test_lib/xrt_test_wrapper.h))</u>
The main changes needed for the host code is declare a buffer object for trace data and pass that buffer object to the XRT kernel function call. This looks like the following snippets of code:

```c
    auto bo_trace = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(7));

    ...

    char *bufTrace = bo_trace.map<char *>();
    memset(bufTrace, 0, myargs.trace_size);
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    ...

    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_in2, bo_out, 0, bo_trace);
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
```
Once the design has been executed. We can then use the convenience function `write_out_trace` to write the buffer contents to a file for post-processing.
```c
    test_utils::write_out_trace((char *)bufTrace, myargs.trace_size, myargs.trace_file);
```

#### Templated host code (test.cpp) 
Because the code patterns for measuring host code timing and configuring trace are so often repeated, they have been further wrapped into the convenience function `setup_and_run_aie` in [xrt_test_wrapper_.h](../../../runtime_lib/test_lib/xrt_test_wrapper.h) which then allows us to create a simpler top level host code [test.cpp](./test.cpp). 

In our template host code [test.cpp](./test.cpp) for 2 inputs and 1 output, we cusotmize the following:
* Input and output buffer size (in bytes) - Specified in the [Makefile](./Makefile) and [CMakeLists.txt](./CMakeLists.txt) and then passed into the [aie2.py](./aie2.py) and [test.cpp](./test.cpp)
    ```Makefile
        in1_size = 16384 # in bytes
        in2_size = 4 # in bytes, should always be 4 (1x int32)
        out_size = 16384 # in bytes, should always be equal to in1_size
    ```
* Buffer data types - Defined in [aie2.py](./aie2.py) and [test.cpp](./test.cpp). The types should match but even if they don't, the buffer size will match and prevent hangs.

    In [aie2.py](./aie2.py):
    ```Python
        in1_dtype = np.int32
        in2_dtype = np.int32
        out_dtype = np.int32
    ```
    In [test.cpp](./test.cpp)
    ```C
        using DATATYPE_IN1 = std::int32_t;
        using DATATYPE_IN2 = std::int32_t;
        using DATATYPE_OUT = std::int32_t;
    ```
* Buffer initialization functions, Verificiation function - Defined in [test.cpp](./test.cpp) and passed into `setup_and_run_aie` as shown below:
    ```C
    // Initialize Input buffer 1
    void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
    for (int i = 0; i < SIZE; i++)
        bufIn1[i] = i + 1;
    }

    // Initialize Input buffer 2
    void initialize_bufIn2(DATATYPE_IN2 *bufIn2, int SIZE) {
    bufIn2[0] = 3; // scaleFactor
    }

    // Initialize Output buffer
    void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
    memset(bufOut, 0, SIZE);
    }

    // Functional correctness verifyer
    int verify_vector_scalar_mul(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                                DATATYPE_OUT *bufOut, int SIZE, int verbosity) {
    int errors = 0;

    for (int i = 0; i < SIZE; i++) {
        int32_t ref = bufIn1[i] * bufIn2[0];
        int32_t test = bufOut[i];
        if (test != ref) {
        if (verbosity >= 1)
            std::cout << "Error in output " << test << " != " << ref << std::endl;
        errors++;
        } else {
        if (verbosity >= 1)
            std::cout << "Correct output " << test << " == " << ref << std::endl;
        }
    }
    return errors;
    }
    ```

In the example simplified `vector_scalar_mul` design, we can build the complete design, including the C/C++ host code [test.cpp](./test.cpp) by running:
```bash
make trace
```

### <u>(2b) Python Host code ([test.py](./test.py), [../../../python/utils/xrt.py](../../../python/utils/xrt.py))</u>
In the [Makefile](./Makefile), we also have a `trace_py` target which calls the python host code `test.py` instead of the C/C++ host code `test.cpp`. 

The python equivalent host code performs the same steps as the C/C++ host code as shown below:
```python
    app.register_buffer(7, shape=trace_buf_shape, dtype=trace_buf_dtype)
    full_output, trace_buffer = execute(
        app, in1_data, in2_data, enable_trace, trace_after_output
    )
```
These convenience python wrappers perform the `sync` steps under the hood when the buffers are being written to and read from. Once the trace data has been written to DDR and read into the trace buffer, we can write it out to a file with the following functions:
```python
    trace_buffer = trace_buffer.view(np.uint32)
    write_out_trace(trace_buffer, str(opts.trace_file))
```
Just like the C/C++ host code wrapper `setup_and_run_aie` found in [../../../runtime_lib/test_lib/xrt_test_wrapper_.h](../../../runtime_lib/test_lib/xrt_test_wrapper.h), for python, we have a similar wrapper `setup_and_run_aie` in [../../../python/utils/xrt.py](../../../python/utils/xrt.py). This likewise simplifies the `test.py` and can be used as a template for design patterns.

## <u>3. Parse text file to generate a waveform json file</u>
Once the packet trace text file is generated (`trace.txt`), we use a python-based trace parser ([parse_trace.py](../../../programming_examples/utils/parse_trace.py)) to interpret the trace values and generate a waveform json file for visualization (with Perfetto). This is a step in the [Makefile](./Makefile) but can be executed from the command line as well.
```Makefile
	../../../programming_examples/utils/parse_trace.py --filename trace.txt --mlir build/aie_trace.mlir --colshift 1 > trace_4b.json
```
This leverages the python parse scripts under [programming_examples/utils](../../../programming_examples/utils/). See the [README.md](../../../programming_examples/utils/README.md) to get more details about how to use the python parse scripts.

In our example [Makefile](./Makefile), we also run [get_trace_summary.py](../../../programming_examples/utils/get_trace_summary.py) to analyze the generated JSON trace file to count the number of invocations of the kernel and the cycle count of those invocations. This depends on the kernel having an `event0` and `event1` function call at the beginning and end of the kernel, which our example does. `event0` and `event1` are functions that generate an internal event and is helpful for us to mark the boundaries of a function call.

## <u>4. Open json file in a visualization tool like Perfetto</u>
Open https://ui.perfetto.dev in your browser and then open up the waveform json file generated in step 3. You can navigate the waveform viewer as you would a standard waveform viewer and can even zoom/pan the waveform with the a,s,w,d keyboard keys.

## <u>Additional Debug Hints</u>
* If you are not getting valid trace data out (e.g. empty `trace.txt` or just 0's), then trace packets were not written to a file successfully. There could be a number of reasons for this but some things to check are:
    * Check if `colshift` is correctly specified (should be correct if called from updated `Makefile`). Phoenix (npu) devices should have `colshift=1` while Strix (npu2) should have `colshift=0`.
    * Did you write to the correct XRT buffer object in your source python that the your host code is reading from. For example, calls to `enable_trace` in high-level IRON python or `configure_packet_tracing_aie2` in closer-to-metal IRON python writes to `ddr_id=4` or `group_id=7` by default. but some other implementations might share the output buffer (`ddr_id=2` or `group_id=5`) so double check which one is being used.
    * It's possible that a simple core may have too few events to create a valid trace packet. To work around this in closer-to-metal IRON python, you can either (1) add a ShimTile to the array of `[tiles_to_trace]` as well to add more trace data or (2) reduce the shim dma burst length by adding the parameter `shim_burst_length=64` to the call `configure_packet_tracing_aie2`. Valid burst shim burst length for aie2 is 64B, 128B, 256B, 512B. The default burst length for regular data buffers is 256-Bytes, but for the trace buffer, it is 64-Bytes instead, which means you only need to define it if it was overwritten elsewhere. This also means that if the trace data is less than 64B, it will not be written out to DDR. Another scenario is that some trace data packets can be missing at the end if it's not am multiple of 64-Bytes.
    * If you're sharing a buffer object for both output and trace, ensure the offset for the trace configuration is the right size (based on output buffer size). Check both size and datatype. Offsets are usually in terms of bytes.
    * Check that the correct tile is being routed to the correct shim DMA. It's not uncommon in a multi core design to route the wrong tile if you're routing these manually, espeically if the tile names might be very similar. Using the convenience python wrappers should automatically handle this correctly.
    * For designs with packet-routing flows, check for correctly matching packet flow IDs. The packet flow ID must match the configured ID value in Trace Control 1 register or else the packets don't get routed. Using the convenience python wrappers should again automatically handle this correctly. However, if your design uses its own packet-routing flows, the default flow IDs may conflict with the trace ones (to be improved in future release)
    
## <u>Exercises</u>
1. Let's give tracing a try. In this directory, we will be examining a simplified version of the `vector scalar multiply` example. Run `make trace`. This compiles the design, generates a trace data file, and runs `prase_trace.py` to generate the `trace_4b.json` waveform file.

    Open this waveform json in http://ui.perfetto.dev. If you zoom into the region of interest with the keyboard shortcut keys W and S to zoom in and out respectively and A and D to pan left and right. You should see a wave like the following:

    <img src="../../assets/trace_vector_scalar_mul1.png" title="AIE-ML Vector Unit." height=250>

    Based on this wave, You can mouse over each chunk of continguous data for `PortRunning0` (input dma port) and `PortRunning1` (output dma port). What is the chunk size? <img src="../../../mlir_tutorials/images/answer1.jpg" title="1024" height=25> How many input and output chunks are there? <img src="../../../mlir_tutorials/images/answer1.jpg" title="4 inputs and 4 outputs (last output might be truncated in viewer)" height=25> This should match iteration loop bounds in our example design.

    There are a few common events in our waveform that are described below:
    * `INSTR_EVENT_0` - The event marking the beginning of our kernel. See [vector_scalar_mul.cc](./vector_scalar_mul.cc) where we added the function `event0()` before the loop. This is generally a handy thing to do to attach an event to the beginning of our kernel.
    * `INSTR_EVENT_1` - The event marking the end of our kernel. See [vector_scalar_mul.cc](./vector_scalar_mul.cc) where we added the function `event1()` after the loop. Much like event0, attaching event1 to the end of our kernel is also helpful.
    * `INSTR_VECTOR` - Vector instructions like vector MAC or vector load/store. Here, we are running a scalar implementation so there are no vector events.
    * `PORT_RUNNING_0` up to `PORT_RUNNING_7` - You can listen for a variety of events, such as `PORT_RUNNING`, `PORT_IDLE` or `PORT_STALLED` on up to 7 ports. To select which port to listen to, use the `PortEvent` Python class as your event. For example, to listen to master port 1:
        ```
        from aie.utils.trace import configure_simple_tracing_aie2, PortEvent
        from aie.utils.trace_events_enum import CoreEvent, MemEvent, PLEvent, MemTileEvent
        trace_utils.configure_simple_tracing_aie2(
            # ... other arguments as above
            events=[trace_utils.PortEvent(CoreEvent.PORT_RUNNING_0, 1, master=True)]
        )
        ```
    * `PORT_RUNNING_1` - Mapped to Port 1 which is by default configured to the MM2S0 output (DMA from local memory to stream). This is usually the first output.
    * `LOCK_STALL` - Any locks stalls.
    * `INSTR_LOCK_ACQUIRE_REQ` - Any lock acquire requests.
    * `INSTR_LOCK_RELEASE_REQ` - Any lock release requests.

    We will look at more exercises with Trace and performance measurement in the next [section](../section-4c).

-----
[[Prev]](../section-4a) [[Up]](../../section-4) [[Next]](../section-4c)
