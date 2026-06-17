<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Section 4b - Trace</ins>

* [Section 4 - Performance Measurement & Vector Programming](../../section-4)
    * [Section 4a - Timers](../section-4a)
    * Section 4b - Trace
    * [Section 4c - Kernel Vectorization and Optimization](../section-4c)

-----

In the previous [section-4a](../section-4a), we looked at how timers can be used to get an overview of application performance. However, for kernel programmers that want to optimize the AIE hardware to its fullest potential, being able to see how efficiently the AIE cores and data movers are running is important. As such, the AIEs are equipped with tracing hardware that provides a cycle accurate view of hardware events. More detailed specification of the AIE2 trace unit can be found at in [AM020](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Trace).

Enabling trace support can be done with the following steps:

## <u>Steps to Enable Trace Support</u>
1. [Enable and configure trace](#1-enable-and-configure-aie-trace)
1. [Configure host code to read trace data and write it to a text file](#2-configure-host-code-to-read-trace-data-and-write-it-to-a-text-file)
1. [Parse text file to generate a waveform json file](#3-parse-text-file-to-generate-a-waveform-json-file)
1. [Open json file in a visualization tool like Perfetto](#4-open-json-file-in-a-visualization-tool-like-perfetto)
* [Additional Debug Hints](#additional-debug-hints)

## <u>1. Enable and configure AIE trace</u>

Enabling tracing means configuring the trace units for a given tile and then routing the generated event packets through the stream switches to the shim DMA where we can write them to a buffer in DDR for post-runtime processing. For IRON, tracing separates two concerns:

* **Sources** — *what* to capture, declared per tile with a `TileTrace`.
* **The sink** — *one* shared output buffer, a `TraceBuffer` passed to `Program(trace=...)`. There is a single buffer; all traced tiles multiplex into it (the hardware distinguishes them by packet id, and `parse_trace` demuxes them host-side).

To trace a compute tile, attach a `TileTrace` to its `Worker`:
```python
worker = Worker(
    core_body,
    fn_args=[of_in.cons(), of_factor.cons(), of_out.prod(), scale],
    trace=TileTrace(),
)
...
return Program(
    iron.get_current_device(),
    rt,
    workers=[worker],
    trace=TraceBuffer(trace_size=trace_size),
).resolve_program()
```
`TileTrace()` with no arguments traces both hardware trace units on the compute tile with sensible defaults (see below). The `TraceBuffer` is the sink that owns the shared DDR buffer.

A compute tile has **two** hardware trace units: the **core** unit (events of type `CoreEvent.*`) and the **core-memory** unit (events of type `MemEvent.*`). When you supply a custom event list, you put BOTH kinds in ONE list — `TileTrace` infers each event's unit from its type and splits them across the two units automatically:
```python
worker = Worker(
    core_body,
    fn_args=[...],
    trace=TileTrace(
        events=[
            CoreEvent.INSTR_EVENT_0,   # core unit
            CoreEvent.INSTR_EVENT_1,   # core unit
            MemEvent.DMA_S2MM_0_START_TASK,  # core-memory unit
        ]
    ),
)
```

A common idiom is to gate tracing on a config so the same generator produces both a traced and an untraced build:
```python
worker = Worker(core_body, fn_args=[...], trace=TileTrace() if trace_config else None)
...
return Program(device, rt, workers=[worker], trace=trace_config).resolve_program()
```
where `trace_config` is a `TraceBuffer` (or `None`). Configuring the trace unit in each traced tile and routing the trace packets to a valid shim tile is then done automatically.

To trace **mem tiles** or **shim tiles** (which are not owned by a Worker), pass `trace_tiles=[...]` to `Program`, each entry a `TileTrace` with an explicit `tile=`:
```python
Program(
    device,
    rt,
    workers=[worker],
    trace_tiles=[
        TileTrace(tile=mem_tile, events=[MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True)]),
        TileTrace(tile=shim_tile, events=[ShimTileEvent.DMA_S2MM_0_START_TASK]),
    ],
    trace=TraceBuffer(trace_size=trace_size),
).resolve_program()
```
Use `MemTileEvent.*` events for a mem tile and `ShimTileEvent.*` events for a shim tile. Full event vocabularies are listed in the [Customizing Trace Behavior](#customizing-trace-behavior) subsection below. The events are imported from `aie.utils.trace.events`, and `TileTrace` / `TraceBuffer` from `aie.utils.trace`:
```python
from aie.iron import TileTrace, TraceBuffer
from aie.utils.trace.events import (
    CoreEvent, MemEvent, MemTileEvent, ShimTileEvent, PortEvent, WireBundle,
)
```
The worked example this section is based on, [aie_trace.py](../../../programming_examples/basic/event_trace/aie_trace.py), uses exactly this pattern: a worker `TileTrace(events=[...])` mixing `CoreEvent` and `MemEvent`, plus `trace_tiles` for a mem tile and a shim tile, with `Program(trace=TraceBuffer(...))`.

### <u>Customizing Trace Behavior</u>

The trace configuration chooses helpful default settings so you can trace your design with little additional customization (just `TileTrace()` and a `TraceBuffer(trace_size=...)`). However, if you want more control, you customize the **sources** by building event lists and passing them to the relevant `TileTrace(events=[...])`, and you customize the **sink** through `TraceBuffer` constructor arguments.

Event vocabularies, by trace unit:
* **core** unit — `CoreEvent.*`. Search under https://xilinx.github.io/mlir-aie/AIEXDialect.html for CoreEvent for the target device [[aie1](https://xilinx.github.io/mlir-aie/AIEXDialect.html#coreeventaie)][[aie2](https://xilinx.github.io/mlir-aie/AIEXDialect.html#coreeventaie2)][[aie2p](https://xilinx.github.io/mlir-aie/AIEXDialect.html#coreeventaie2p)].
* **core-memory** unit — `MemEvent.*`. Search under https://xilinx.github.io/mlir-aie/AIEXDialect.html for MemEvent for the target device [[aie1](https://xilinx.github.io/mlir-aie/AIEXDialect.html#coreeventaie)][[aie2](https://xilinx.github.io/mlir-aie/AIEXDialect.html#coreeventaie2)][[aie2p](https://xilinx.github.io/mlir-aie/AIEXDialect.html#coreeventaie2p)].
* **mem tile** unit — `MemTileEvent.*`. Search under https://xilinx.github.io/mlir-aie/AIEXDialect.html for MemTileEvent for the target device [[aie1](https://xilinx.github.io/mlir-aie/AIEXDialect.html#memevent)][[aie2](https://xilinx.github.io/mlir-aie/AIEXDialect.html#memevent2)][[aie2p](https://xilinx.github.io/mlir-aie/AIEXDialect.html#memevent2p)]
* **shim tile** unit — `ShimTileEvent.*`. Search under https://xilinx.github.io/mlir-aie/AIEXDialect.html for ShimTileEvent for the target device [[aie1](https://xilinx.github.io/mlir-aie/AIEXDialect.html#shimtileevent)][[aie2](https://xilinx.github.io/mlir-aie/AIEXDialect.html#shimtileevent2)][[aie2p](https://xilinx.github.io/mlir-aie/AIEXDialect.html#shimtileevent2p)]

For a worker's compute tile, mix the `CoreEvent.*` (core unit) and `MemEvent.*` (core-memory unit) events you care about into a single list — `TileTrace` infers each event's unit from its type and splits them automatically. Each hardware unit provides 8 event slots:

```python
from aie.iron import TileTrace
from aie.utils.trace.events import CoreEvent, MemEvent

worker = Worker(
    core_body,
    fn_args=[...],
    trace=TileTrace(
        events=[
            CoreEvent.INSTR_EVENT_0,
            CoreEvent.INSTR_EVENT_1,
            CoreEvent.INSTR_VECTOR,
            CoreEvent.MEMORY_STALL,
            CoreEvent.STREAM_STALL,
            CoreEvent.LOCK_STALL,
            MemEvent.DMA_S2MM_0_START_TASK,
            MemEvent.DMA_MM2S_0_FINISHED_TASK,
        ]
    ),
)
```

For a mem tile or shim tile, build a `TileTrace(tile=..., events=[...])` (with `MemTileEvent.*` or `ShimTileEvent.*` respectively) and pass it in `Program(trace_tiles=[...])`, as shown in [section 1](#1-enable-and-configure-aie-trace) above.

The sink is customized through `TraceBuffer` constructor arguments:
* `trace_size` - size of the shared trace buffer in bytes.
* `ddr_id` - XRT buffer index (0-4) to write trace data to, mapping to group_id (3-7). Defaults to 4 (group_id 7). Set to -1 to append trace data after the last runtime_sequence tensor argument. See [below](#2-configure-host-code-to-read-trace-data-and-write-it-to-a-text-file) for more details on XRT buffers.
* `trace_file` - host file the trace buffer is written to (default `trace.txt`).
* `egress_shim_col` - shim column used to egress trace data to DDR.

```python
from aie.iron import TraceBuffer

Program(
    device,
    rt,
    workers=[worker],
    trace=TraceBuffer(trace_size=trace_size, ddr_id=4),
).resolve_program()
```

### <u>PortEvent API</u>

Port events monitor activity on stream switch ports. A **physical port** on the stream switch is identified by three components: the **bundle** (which interface on the tile, e.g., DMA, North, South), the **channel** (which channel within that bundle), and the **direction** (master for input/S2MM, slave for output/MM2S).

The hardware provides 8 port monitor slots (0-7), each configured to watch a specific physical port. The slot number is determined by the suffix of the event name (e.g., `PORT_RUNNING_0` uses slot 0).

```python
from aie.utils.trace.events import PortEvent, MemTilePortEvent, ShimTilePortEvent, CoreEvent, MemTileEvent, ShimTileEvent
from aie.dialects.aie import WireBundle

# Core tile port monitoring
PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True)

# Parameters:
# - code: PORT_RUNNING_N, PORT_IDLE_N, PORT_STALLED_N, or PORT_TLAST_N (N=0-7)
# - bundle: WireBundle.DMA, WireBundle.North, WireBundle.South, etc.
# - channel: Channel number within the bundle (e.g., 0 for DMA channel 0)
# - master: Direction - True for input to tile (S2MM), False for output from tile (MM2S)

# Mem tile port monitoring
MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, WireBundle.DMA, channel=0, master=True)

# Shim tile port monitoring
ShimTilePortEvent(ShimTileEvent.PORT_RUNNING_0, WireBundle.South, channel=2, master=True)
```

**Sharing a monitor slot across event types:** Multiple event types can share the same slot to observe different conditions on the same port. The event name suffix determines the slot number (`PORT_RUNNING_0` and `PORT_TLAST_0` both use slot 0). When events share a slot, each event type independently triggers in the trace whenever its condition is met on the monitored port — `PORT_RUNNING_0` fires while data is flowing and `PORT_TLAST_0` fires on the last beat of a transfer. They must be configured to monitor the same physical port or an error is raised.

## <u>2. Configure host code to read trace data and write it to a text file</u>

Once the trace units are configured and routed, we want the host code to read the trace data from DDR and write it out to a text file for post-run processing. To give a better sense of how this comes together, this section provides an example design that is again a simplifed version of the [Vector Scalar Multiply example](../../../programming_examples/basic/vector_scalar_mul/).

### <u>AIE structural design code ([vector_scalar_mul.py](./vector_scalar_mul.py))</u>
In order to write the DDR data to a text file, we need to know where in DDR the trace data is stored and then read from that location. This starts inside the [vector_scalar_mul.py](./vector_scalar_mul.py) file where the `TraceBuffer` sink (together with the per-tile `TileTrace` sources) under the hood expands to calls to configure the trace units and program the shimDMA to write to one of XRT inout buffers. It is helpful to have a more in-depth understanding about the *XRT buffer objects* described in [section 3](../../section-3). There we had described that our XRT supports up to 5 inout buffer objects. Common usage patterns include 1 input/ 1 output and 2 input/ 1 output. These patterns then map in the following way where the *group_id* is listed next to each XRT buffer object, `inoutN (group_id)`.

| inout0 (3) | inout1 (4) |
|--------|--------|
| input A  | output C |

| inout0 (3) | inout1 (4) | inout2 (5) |
|--------|--------|--------|
| input A  | input B | output C  |

To support trace, we will configure a shim tile to move the trace packet data to DDR through one of these XRT buffer objects. For simplicity, we choose `inout4 (7)` as the default case such that the new trace enabled mapping is:

| inout0 (3)| inout1 (4) | inout2 (5) | inout3 (6)| inout4 (7)|
|--------|--------|--------|--------|--------|
| input A  | output C | unused | unused | trace  |

| inout0 (3)| inout1 (4)| inout2 (5)| inout3 (6)| inout4 (7)|
|--------|--------|--------|--------|--------|
| input A  | input B | output C | unused | trace  |

In some designs, we have also used a pattern where we share an XRT buffer object where the trace data is written to same buffer object as the output by setting `ddr_id=-1`. This is helpful if we do not have a spare buffer object dedicated to trace, but requires precise declaration of offset size. See [Conv2d example](../../../programming_examples/ml/conv2d/).

| inout0 (3)| inout1 (4)| inout2 (5)|
|--------|--------|--------|
| input A  | input B | (output C + trace) |

By specifying `inout4 (7)` as the default case, we can leave the parameters for `TraceBuffer` to their default values other than `trace_size`. However, if we do decide to customize the XRT buffer object used, we can do so through `TraceBuffer(ddr_id=...)` (to specify the buffer to use). Setting `ddr_id=-1` appends trace data after the last output tensor, using the last argument's buffer index and a byte offset equal to the tensor size.

Once the design is configured to a XRT buffer object, we turn our attention to the host code to read the DDR data and write it to a file.

> **NOTE** In our example design, we provide a [Makefile](./Makefile) target `run` for standard build and `trace` for trace-enabled build. The trace-enabled build passes the trace buffer size as an argument which is used under the hood to conditionally enable tracing as long as `trace_size` is > 0. This is also true for the [Vector Scalar Multiply example](../../../programming_examples/basic/vector_scalar_mul).

### <u>(2a) C/C++ Host code ([test.cpp](./test.cpp), [../../../runtime_lib/test_lib/xrt_test_wrapper.h](../../../runtime_lib/test_lib/xrt_test_wrapper.h))</u>
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
Because the code patterns for measuring host code timing and configuring trace are so often repeated, they have been further wrapped into the convenience function `setup_and_run_aie` in [xrt_test_wrapper.h](../../../runtime_lib/test_lib/xrt_test_wrapper.h) which then allows us to create a simpler top level host code [test.cpp](./test.cpp).

In our template host code [test.cpp](./test.cpp) for 2 inputs and 1 output, we customize the following:
* Input and output buffer size (in bytes) - Specified in the [Makefile](./Makefile) and [CMakeLists.txt](./CMakeLists.txt) and then passed into the [vector_scalar_mul.py](./vector_scalar_mul.py) and [test.cpp](./test.cpp)
    ```Makefile
        in1_size = 16384 # in bytes
        in2_size = 4 # in bytes, should always be 4 (1x int32)
        out_size = 16384 # in bytes, should always be equal to in1_size
    ```
* Buffer data types - Defined in [vector_scalar_mul.py](./vector_scalar_mul.py) and [test.cpp](./test.cpp). The types should match but even if they don't, the buffer size will match and prevent hangs.

    In [vector_scalar_mul.py](./vector_scalar_mul.py):
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

* Setup and run program - The function wrapper `setup_and_run_aie` then sets up the device and XRT buffers and runs the program as defined within [../../../runtime_lib/test_lib/xrt_test_wrapper.h](../../../runtime_lib/test_lib/xrt_test_wrapper.h). Here, we see that `setup_and_run_aie` also handles the trace configuration, trace buffer setup and synchronization, and writing trace data to an output file.

In the example simplified `vector_scalar_mul` design, we can build the complete design, including the C/C++ host code [test.cpp](./test.cpp) by running:
```bash
make trace
```


### <u>(2b) Python Host code ([test.py](./test.py), [../../../python/xrt.py](../../../python/xrt.py))</u>
In the [Makefile](./Makefile), we also have a `trace_py` target which calls the python host code `test.py` instead of the C/C++ host code `test.cpp`.

#### test_utils (recommended)

The recommended approach is to use `test_utils.create_npu_kernel`, which creates both a [`TraceBuffer`](../../../python/utils/trace/config.py) and an [`NPUKernel`](../../../python/utils/npukernel.py) from command-line arguments:

```python
import aie.utils.test as test_utils
...
npu_opts = test_utils.create_npu_kernel(opts)
res = DefaultNPURuntime.run_test(npu_opts.npu_kernel, ...)
```

The relevant CLI arguments (added by `aie.utils.hostruntime.argparse.add_runtime_args`) are:
- `-t` / `--trace_size`: Trace buffer size in bytes. Tracing is enabled when this is > 0.
- `--trace-file`: Path to write raw trace data (default: `trace.txt`).
- `--ddr-id`: DDR buffer index for trace (0-4, or -1 to append after last tensor). Default is 4.

> **IMPORTANT**: The `ddr_id` value (set via `--ddr-id`) **must match** the `ddr_id` parameter of the `TraceBuffer` in your IRON design (`TraceBuffer(ddr_id=...)`), or buffer allocation will be incorrect.

#### TraceBuffer (manual setup)

For custom host code, you can create a [`TraceBuffer`](../../../python/utils/trace/config.py) directly and pass it to [`NPUKernel`](../../../python/utils/npukernel.py):

```python
from aie.iron import TraceBuffer
from aie.utils.npukernel import NPUKernel

trace_config = TraceBuffer(
    trace_size=8192,                # Buffer size in bytes
    trace_file="trace.txt",         # Output file for raw trace data (default)
)

npu_kernel = NPUKernel(
    xclbin_path="build/final.xclbin",
    insts_path="build/insts.txt",
    trace_config=trace_config,
)
```

Under the hood, the [DefaultNPURuntime](../../../python/utils/hostruntime/hostruntime.py) uses `TraceBuffer` to allocate the trace XRT buffer, synchronize it after execution, and write the trace data to the output file -- similar to the C++ `write_out_trace` function and `setup_and_run_aie` wrapper in [xrt_test_wrapper.h](../../../runtime_lib/test_lib/xrt_test_wrapper.h).

#### `@iron.jit` integration

For designs using `@iron.jit`, `TraceBuffer` can travel as a
`CompileTime[T]`-style argument and be evaluated **inside** the generator,
so the trace-vs-no-trace decision lives in the design itself instead
of the host:

```python
from aie.iron import TileTrace, TraceBuffer

@iron.jit
def passthrough_with_trace(
    x: In,
    y: Out,
    *,
    N: CompileTime[int],
    trace_config: CompileTime[TraceBuffer | None] = None,
):
    line_size = N // 4
    line_ty = np.ndarray[(line_size,), np.dtype[np.uint8]]
    of_in = ObjectFifo(line_ty, name="in")
    of_out = ObjectFifo(line_ty, name="out")
    pt = kernels.passthrough(tile_size=line_size, dtype=np.uint8)

    def core_fn(of_in, of_out, pt):
        for _ in range_(4):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            pt(elem_in, elem_out, line_size)
            of_in.release(1)
            of_out.release(1)

    # Enable per-worker trace instrumentation only when a config is bound.
    worker = Worker(
        core_fn, [of_in.cons(), of_out.prod(), pt],
        trace=TileTrace() if trace_config else None,
    )

    rt = Runtime()
    tensor_ty = np.ndarray[(N,), np.dtype[np.uint8]]

    def sequence(a_in, b_out):
        of_in.prod().fill(a_in)
        of_out.cons().drain(b_out, wait=True)

    rt.sequence(sequence, [tensor_ty, tensor_ty])

    # The TraceBuffer sink is passed to Program(trace=...). When trace_config
    # is None, the trace plumbing is omitted from the lowered MLIR entirely.
    return Program(
        iron.get_current_device(),
        rt,
        workers=[worker],
        trace=trace_config,
    ).resolve_program()
```

Two equivalent ways to drive it from the caller:

```python
# No trace — trace_config defaults to None, so trace plumbing is omitted
# from the lowered MLIR entirely (clean cache hit on the same recipe).
passthrough_with_trace(in_t, out_t, N=4096)

# With trace — pass a TraceBuffer, then read parsed output back out.
trace_cfg = TraceBuffer(trace_size=8192)
passthrough_with_trace(in_t, out_t, N=4096, trace_config=trace_cfg)
trace_cfg.trace_to_json(trace_cfg.physical_mlir_path, "trace.json")
```

Two side effects of the call worth knowing about:

- `trace_config.physical_mlir_path` is auto-populated with the
  per-design `input_with_addresses.mlir` path (under
  `$NPU_CACHE_HOME/<hash>/` — see
  [compilation_stages.md](../../compilation_stages.md) §Lowering).
  Pass it straight into `trace_config.trace_to_json(mlir, "out.json")`
  to parse `trace.txt` into Chrome-tracing JSON without locating the
  cache dir yourself.
- `aie.utils.trace.utils.print_cycles_summary(json_path)` walks the
  generated JSON and prints per-event cycle counts.  Most useful with
  kernels whose C++ body brackets work with `event0()` / `event1()`
  (the `kernels.*` library helpers do) so the summary can pair entry
  and exit timestamps.

## <u>3. Parse text file to generate a waveform json file</u>
Once the packet trace text file is generated (`trace.txt`), we use a python-based trace parser ([parse.py](../../../python/utils/trace/parse.py)) to interpret the trace values and generate a waveform json file for visualization (with Perfetto). This is a step in the [Makefile](./Makefile) but can be executed from the command line as well.

The `--mlir` argument should point to `input_with_addresses.mlir` from the `.prj` work directory, not the original source MLIR. This file contains the lowered register writes produced by the trace passes, which the parser uses to map raw trace packets back to named events.

```bash
python ../../../python/utils/trace/parse.py \
    --input trace.txt \
    --mlir build/aie.mlir.prj/input_with_addresses.mlir \
    --output trace.json
```

In our example [Makefile](./Makefile), we also run [get_trace_summary.py](../../../python/utils/trace/get_trace_summary.py) to analyze the generated JSON trace file to count the number of invocations of the kernel and the cycle count of those invocations. This depends on the kernel having an `event0` and `event1` function call at the beginning and end of the kernel, which our example does. `event0` and `event1` are functions that generate an internal event and is helpful for us to mark the boundaries of a function call.

## <u>4. Open json file in a visualization tool like Perfetto</u>
Open https://ui.perfetto.dev in your browser and then open up the waveform json file generated in step 3. You can navigate the waveform viewer as you would a standard waveform viewer and can even zoom/pan the waveform with the a,s,w,d keyboard keys.

## <u>Additional Debug Hints</u>
* If you are not getting valid trace data out (e.g. empty `trace.txt` or just 0's), then trace packets were not written to a file successfully. There could be a number of reasons for this but some things to check are:
    * Did you write to the correct XRT buffer object that your host code is reading from? The default is `ddr_id=4` (`group_id=7`), which means trace data is written to a dedicated XRT buffer. If using `ddr_id=-1`, trace data is appended after the last tensor argument.
        * If using the **Python host** (`DefaultNPURuntime` / `TraceBuffer`), buffer management is handled automatically. However, the `ddr_id` in the host `TraceBuffer` must match the `ddr_id` of the `TraceBuffer` passed to `Program(trace=...)` in your IRON design.
        * If using a **C/C++ host** with `ddr_id=-1`, trace data is appended to the last `runtime_sequence` argument's buffer at an offset equal to the output size. Allocate that buffer large enough for both output and trace data, and do **not** create a separate `bo_trace` at `group_id(7)`.
    * It's possible that a simple core may have too few events to create a valid trace packet. For dialect-level designs, you can work around this by adding a ShimTile to the `tiles_to_trace` array in `configure_trace()` to generate additional trace data.
    * Check that the correct tile is being routed to the correct shim DMA. Using the declarative trace API handles this automatically.
    * You may get an invalid tile error if the `colshift` doesn't match the actually starting column of the design. This should automatically be set by the `parse.py` script but can also be specified manually. Phoenix (npu) devices should have `colshift=1` while Strix (npu2) should have `colshift=0` when allocated to an unused NPU.
    * For designs with packet-routing flows, check for correctly matching packet flow IDs. The packet flow ID must match the configured ID value in Trace Control 1 register or else the packets don't get routed. Using the declarative trace API handles this automatically.

## <u>Exercises</u>
1. Let's give tracing a try. In this directory, we will be examining a simplified version of the `vector scalar multiply` example. Run `make trace`. This compiles the design, generates a trace data file, and runs `parse.py` to generate the `trace_4b.json` waveform file.

    Open this waveform json in http://ui.perfetto.dev. If you zoom into the region of interest with the keyboard shortcut keys W and S to zoom in and out respectively and A and D to pan left and right. You should see a wave like the following:

    <img src="../../assets/trace_vector_scalar_mul1.png" title="AIE-ML Vector Unit." height=250>

    Based on this wave, You can mouse over each chunk of continguous data for `PortRunning0` (input dma port) and `PortRunning1` (output dma port). What is the chunk size? <img src="../../../mlir_exercises/images/answer1.jpg" title="1024" height=25> How many input and output chunks are there? <img src="../../../mlir_exercises/images/answer1.jpg" title="4 inputs and 4 outputs (last output might be truncated in viewer)" height=25> This should match iteration loop bounds in our example design.

    There are a few common events in our waveform that are described below:
    * `INSTR_EVENT_0` - The event marking the beginning of our kernel. See [vector_scalar_mul.cc](./vector_scalar_mul.cc) where we added the function `event0()` before the loop. This is generally a handy thing to do to attach an event to the beginning of our kernel.
    * `INSTR_EVENT_1` - The event marking the end of our kernel. See [vector_scalar_mul.cc](./vector_scalar_mul.cc) where we added the function `event1()` after the loop. Much like event0, attaching event1 to the end of our kernel is also helpful.
    * `INSTR_VECTOR` - Vector instructions like vector MAC or vector load/store. Here, we are running a scalar implementation so there are no vector events.
    * `PORT_RUNNING_0` up to `PORT_RUNNING_7` - You can listen for a variety of events, such as `PORT_RUNNING`, `PORT_IDLE` or `PORT_STALLED` on up to 8 ports. To select which port to listen to, use the `PortEvent` Python class. See [PortEvent API](#portevent-api) for the full `PortEvent` API and examples.
    * `PORT_RUNNING_1` - Mapped to Port 1 which is configured to the MM2S0 output (DMA from local memory to stream) in this example. This is usually the first output based on routing algorithm.
    * `LOCK_STALL` - Any locks stalls.
    * `INSTR_LOCK_ACQUIRE_REQ` - Any lock acquire requests.
    * `INSTR_LOCK_RELEASE_REQ` - Any lock release requests.

    We will look at more exercises with Trace and performance measurement in the next [section](../section-4c).

-----
[[Prev]](../section-4a) [[Up]](../../section-4) [[Next]](../section-4c)
