<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Python Utilities

The python utilties are designed to simplify commonly repeated tasks and wrap them up into helper functions. They are divided into separate categories and can be added to any python code via:
```
import aie.utils.trace as trace_utils
import aie.utils.test as test_utils
```
Thereafter, functions defined in the file can be called via `trace_utils.configure_simple_tracing_aie2(...)`.

- [Test utilities](#test-utilites-testpy) ([test.py](./test.py))
- [Trace utilities](#trace-utilites-tracepy) ([trace.py](./trace.py))
- [XRT utilities](#xrt-utilites-xrtpy) ([xrt.py](./xrt.py))
- [Machine Learning (ML) utilities](#machine-language-ml-utilites-mlpyss) ([ml.py](./ml.py))

## Test utilites ([test.py](./test.py))
Test/ Host code utilities.
* `create_default_argparser`
    * This creates a ArgumentParser with the following args: --xclbin, --kernel, --instr, -v, --verify, --iters, --warmup, --trace_sz, --trace_file
    * It returns the ArgumentParser which allows you to add more arguments
* `parse_args` 
    * Calls create_default_argparser and returns the parsed results
    * Useful if you don't need additional custom args
* `init_xrt_load_kernel`
    * Helpful wrapper for a number of commonly used XRT calls in `test.py`
        * Declare an XRT `device`
        * Load the xclbin file and register the `xclbin`
        * Declare hardware context and use that to return the `device` and `kernel`


## Trace utilites ([trace.py](./trace.py))

* `extract_trace`
    * Used in some jupyter notebook python examples. Given the output buffer, its shape and dtype and the trace_size, it returns the output buffer only (as output_prefix) and the trace buffer (as trace_suffix)
    * However, the process of extracting the output_buffer and trace_buffer can also be as simple as:
        ```python
        entire_buffer = bo_inout1.read(OUT_SIZE, 0).view(np.uint32)
        output_buffer = entire_buffer[:INOUT1_VOLUME]
        trace_buffer = entire_buffer[INOUT1_VOLUME:]
        ```
* `write_out_trace`
    * Write the trace_buffer `trace` to an output file named `file_name`
    * The trace buffer should be stored as one uint32_t per indices/line
* `pack4bytes`
    * Pack 4 bytes into a 32-bit word
* `configure_packet_tracing_aie2` (packet switched multi-tile tracing)
    * This function abstracts a number of functions for configuring multiple core tiles and an associated shim tile. It does not define trace packet routing between them, which `configure_packet_tracing_flow` does. These two functions should be used together.
        
        Function arguments:
        * `tiles to trace` - array of tiles to trace
        * `shim tile` - Single shim tile to configure for writing trace packets to DDR
        * `size` - trace buffer size (in bytes)
        * `offset` - offest (in bytes) where trace buffer data should begin

        An example use case would be:
        ```python
        trace_utils.configure_packet_tracing_aie2(tile_to_trace, ShimTile, opts.trace_size, 4096 * 4)
        ```
* `configure_packet_tracing_flow` (packet switched flows)
    * This function automates the declaration of packet flows for an array of tiles to trace to the target shim tile. Note this function makes assumptions about packet routing IDs and this needs to match on shim config side which is done with `configure_packet_tracing_aie`.         
        
        Function arguments:
        * `tiles to trace` - array of tiles to trace
        * `shim tile` - Single shim tile to configure for writing trace packets to DDR

        An example use case would be:
        ```python
        trace_utils.configure_packet_tracing_flows(tile_to_trace, ShimTile)
        ```

* `configure_simple_tracing_aie2` (cicuit switched single tile tracing)
    * This function abstracts a number of python functions for configuring a core tile and an associated shim tile. It does not define the trace packet routing between the two however. 

        Function arguments:
        * `channel` - S2MM channel used
        * `bd_id` - DMA bd used. Be careful that we do not conflict with the auto-assigned bds from allocated by `npu_dma_memcpy_nd` calls
        * `ddr_id` - Maps to one of the 3 inout buffers (1,2,3)
        * `size` - trace buffer size (in bytes)
        * `offset`- offset (in bytes) where trace buffer data should begin
        * `start`- start event
        * `stop`- stop event
        * `events`- Vector of up to 8 events that we are tracing; these can be any from the `trace_events_enum` described below

        The minimum function call supported is:
        ```python
        trace_utils.configure_simple_tracing_aie2(tile, shim)
        ```
        This version allows the default argument values as described below:
        * `channel`=1 - to configure S2MM channel 1
        * `bd_id`=13 - 13 is far enough that's unlikely to have conflict
        * `ddr_id`=2 - Maps to inout2 buffer
        * `size`=8192 - 8,192 bytes for trace buffer size
        * `offset`=0 - An offset=0 means the trace data is in its own inout buffer (not appended to another channel)
        * `start`=0x1 - Start event triggers right away when tile is enabled
        * `stop`=0x0 - No Stop event
        * `events` - a standard template of events commonly used below as below:
           ```
           events=[ CoreEvent.INSTR_EVENT_1,
                    CoreEvent.INSTR_EVENT_0,
                    CoreEvent.INSTR_VECTOR,
                    CoreEvent.INSTR_LOCK_RELEASE_REQ,
                    CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
                    CoreEvent.LOCK_STALL,
                    PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
                    PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),  # slave(1)
                   ]
           ``` 

        A more common use case might be:
        ```python
        trace_utils.configure_simple_tracing_aie2(tile, shim, size=8192, offset=output_size, ddr_id_=2)
        ```
        This one allows us to control the size, offset, and inout buffer mapping.

        To better appreciate what this wrapper function does, we need to delve more deeply into the details on how trace units are configured.
* Additional helper functions can be found in the `trace.py` and are documented in the source directly.

### Available Events for Tracing - `trace_events_enum.py`

`trace_events_enum.py` contains a list of all traceable events on AIE-ML devices.
These include events on compute cores (`CoreEvent`), memory modules (`MemEvent`), programmable logic (`PLEvent`) and mem tiles (`MemTileEvent`).
When specifying a list of events to `configure_simple_tracing_aie2`, you can refer to events either by their name or their numeric values:
```
configure_simple_tracing_aie2(..., events=[0x4B, 0x22, 0x21, 0x25])
# or, equivalently:
configure_simple_tracing_aie2(..., events=[CoreEvent.INSTR_EVENT_1, CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_VECTOR, CoreEvent.INSTR_LOCK_RELEASE_REQ])
```

#### Port Events

There is a set of events that fire on certain activity on data memory ports.
These are `PORT_IDLE_0` through `PORT_IDLE_7`, `PORT_RUNNING_0` through `PORT_RUNNING_7`, `PORT_STALLED_0` throught `PORT_STALLED_7` and finally `PORT_TLAST_0` through `PORT_TLAST_7`.
You have to specify on which port the tracing engine should listen for each those events.
In hardware, this is done by configuring registers `0x3FF00` and `0x3FF04`.
The Python tracing utilities abstract this in `configure_simple_tracing_aie2`; you only have to specify the event as a `PortEvent` along with the corresponding port as follows:

```
configure_tracing_aie2(
    ...,
    events=[
        PortEvent(CoreEvent.PORT_RUNNING_0, 1, master=True)
        # This will emit an event whenever master port 1 is running.
    ]
)
```

`PortEvent` is defined in `aie.utils.trace` and `CoreEvent` is defined in `aie.utils.trace_events_enum`.

### Configure tile trace settings
Within the `aiex.runtime_sequence` block, we call a set of configuration register writes (`aiex.npu.write32`) to configure the tile trace units and (`aiex.npu.writebd`) to configure the shimDMA. 

For a give AIE2 tile, we configure the trace control registers for the tile core and tile memory separately. There are 4 registers we generally use to configure the trace unit behavior. 2 are for configuring the general trace control and the other 2 are to specify which events our tile's trace hardware is monitoring.

AIE2 core module registers can be found in [AM025](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/).
The table below describes the trace control registers for the core module.

| Config Register | Address | Field | Bits | Reset Value | Description |
|-----------------|---------|-------|------|-------------|-------------|
| Trace Control 0 | 0x340D0 | Stop Event | [30:24], 0xNN------ | 0 | Event to stop trace capture | 
| Trace Control 0 | 0x340D0 | Start Event | [22:16], 0x--NN---- | 0 | Event to start trace capture |
| Trace Control 0 | 0x340D0 | Mode | [1:0], 0x-------N | 0 | Trace mode. 00=event-time, 01=event-PC, 10=execution |
| Trace Control 1 | 0x340D4 | Packet Type | [14:12], 0x----N--- | 0 | Detination trace packet - packet type |
| Trace Control 1 | 0x340D4 | Packet ID | [4:0], 0x------NN | 0 | Detination trace packet - packet ID |

This info is also found online in [AM025](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/) for [Trace Control 0](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/Trace_Control0-CORE_MODULE-Register) and [Trace Control 1](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/Trace_Control1-CORE_MODULE-Register).

**Note** that Trace Control 0 is all you need for circuit switched flows. For packet switched flows, however, you will also need to configure Trace Control 1. Here, the packet type matches the types we define to support post-run parsing. Here is a table of trace unit types and packet type.

| AIE trace unit | packet type |
|-----------------|-------------|
| Tile core       | 0           |
| Tile memory     | 1           |
| ShimTile        | 2           |
| MemTile         | 3           |

<u>Example Trace Control 0 Config</u>

in C/C++
```c++
// Start event = 1, Stop event = 0, Mode = event-time
aiex.npu.write32 { column = 0 : i32, row = 4 : i32, address = 0x340D0 : ui32, value = 0x10000 : ui32 }
aiex.npu.write32 { column = 0 : i32, row = 4 : i32, address = 0x340D4 : ui32, value = 0x0 : ui32 }
```
in Python
```python
# Start event = 1, Stop event = 0, Mode = event-time
npu_write32(column=0, row=4, address=0x340D0, value=pack4bytes(stop, start, 0, 0),)
npu_write32(column=0, row=4, address=0x340D4, value=0,)
```

The table below describes which events the trace hardware monitors.

| Config Register | Address | Field | Bits | Reset Value | Description |
|-----------------|---------|-------|------|-------------|-------------|
| Trace Event Group 1 | 0x340E4 | Trace Event 7 | [30:24], 0xNN------ | 0 | 8th trace event to monitor |
| Trace Event Group 1 | 0x340E4 | Trace Event 6 | [22:16], 0x--NN---- | 0 | 7th trace event to monitor |
| Trace Event Group 1 | 0x340E4 | Trace Event 5 | [14:8], 0x----NN-- | 0 | 6th trace event to monitor |
| Trace Event Group 1 | 0x340E4 | Trace Event 4 | [6:0], 0x------NN | 0 | 5th trace event to monitor |
| Trace Event Group 0 | 0x340E0 | Trace Event 3 | [30:24], 0xNN------ | 0 | 4th trace event to monitor |
| Trace Event Group 0 | 0x340E0 | Trace Event 2 | [22:16], 0x--NN---- | 0 | 3rd trace event to monitor |
| Trace Event Group 0 | 0x340E0 | Trace Event 1 | [14:8], 0x----NN-- | 0 | 2nd trace event to monitor |
| Trace Event Group 0 | 0x340E0 | Trace Event 0 | [6:0], 0x------NN | 0 | 1st trace event to monitor |

This info is also found online in [AM025](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/) for [Trace Event 0](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/Trace_Event0-CORE_MODULE-Register) and [Trace Event 1](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/Trace_Event1-CORE_MODULE-Register).


There is an extensive lists of trace events, but we will only list a few common ones here.
| Some common events | event ID | dec value |
|--------------------|----------|-----------|
| True                       |0x01| 1 |
| Stream stalls              |0x18| 24 |
| Core Instruction - Event 0  |0x21| 33|
| Core Instruction - Event 1  |0x22| 34 |
| Vector Instructions (e.g. VMAC, VADD, VCMP) |0x25|  37 |
| Lock acquire requests      |0x2C|  44 |
| Lock release requests      |0x2D|  45 | 
| Lock stall                 |0x1A|  26 |
| Core Port Running 1        |0x4F|  79 |
| Core Port Running 0        |0x4B|  75 | 
* A more exhaustive list of events for core tile, core memory, memtile and shim tile can be found in this [header file](https://github.com/Xilinx/aie-rt/blob/main-aie/driver/src/events/xaie_events_aie.h). However, not all events are yet supported in `parse_eventIR.py` at this time.

**NOTE**: The "Core Instruction - Event 0/1" are special intrinsics you can add to your kernel code to trigger an event during the running of your core program. Within the kernel code, they look like:
```c++
event0();
...
event1();
```
This can be placed at the beginning and end of your code block to estimate the total execution time of your kernel program.

<u>Example Trace Events 0/1 Config</u>

Setting the trace events registers can again be done using the `aiex.npu.write32` function targeting the correct register address (0x340E0 and 0x340E4 for core tiles). An example of setting them in your host code is below:

in C/C++
```c++
// Events 0-3 monitored
// ------------------------
// Vector instrucitons (0x25)
// Core Instruction - Event 0 (0x21)
// Core Instruction - Event 1 (0x22)
// Core Port Running 0 (0x4B) 
aiex.npu.write32 { column = 0 : i32, row = 4 : i32, address = 0x340E0 : ui32, value = 0x4B222125 : ui32 }

// Events 4-7 monitored
// ------------------------
// Core Port Running 1 (0x4F)
// Lock stalls (0x1A)
// Lock acquire requests (0x2C)
// Lock release requests (0x2D)
aiex.npu.write32 { column = 0 : i32, row = 4 : i32, address = 0x340E4 : ui32, value = 0x2D2C1A4F : ui32 }
```
in Python
```python
# events=[0x4B, 0x22, 0x21, 0x25, 0x2D, 0x2C, 0x1A, 0x4F]
npu_write32(column=0, row=4, address=0x340E0, value=*events[0:4],)
npu_write32(column=0, row=4, address=0x340E4, value=*events[4:8],)
```

Some configurations like the Port Running 0/1 events are further configured by a secondary configuration register. In this case, we route the port activity from the stream switch to Port running 0 or 1. 
| Config Register | Address | Field | Bits | Reset Value | Description |
|-----------------|---------|-------|------|-------------|-------------|
| Stream Switch Event Port Selection 1 | 0x3FF04 | Port 7 Master/Slave | [29], 0xN------- | 0 | Master or slave for port 7, 1=master, 0=slave |
| Stream Switch Event Port Selection 1 | 0x3FF04 | Port 7 ID | [28:24], 0xNN------ | 0 | Select port ID for port 7 event gen |
| Stream Switch Event Port Selection 1 | 0x3FF04 | Port 6 Master/Slave | [21], 0x--N----- | 0 | Master or slave for port 6, 1=master, 0=slave |
| Stream Switch Event Port Selection 1 | 0x3FF04 | Port 6 ID | [20:16], 0x--NN---- | 0 | Select port ID for port 6 event gen |
| Stream Switch Event Port Selection 1 | 0x3FF04 | Port 5 Master/Slave | [13], 0x----N--- | 0 | Master or slave for port 5, 1=master, 0=slave |
| Stream Switch Event Port Selection 1 | 0x3FF04 | Port 5 ID | [12:8], 0x----NN-- | 0 | Select port ID for port 5 event gen |
| Stream Switch Event Port Selection 1 | 0x3FF04 | Port 4 Master/Slave | [5], 0x------N- | 0 | Master or slave for port 4, 1=master, 0=slave |
| Stream Switch Event Port Selection 1 | 0x3FF04 | Port 4 ID | [4:0], 0x------NN | 0 | Select port ID for port 4 event gen |
| Stream Switch Event Port Selection 0 | 0x3FF00 | Port 3 Master/Slave | [29], 0xN------- | 0 | Master or slave for port 3, 1=master, 0=slave |
| Stream Switch Event Port Selection 0 | 0x3FF00 | Port 3 ID | [28:24], 0xNN------ | 0 | Select port ID for port 3 event gen |
| Stream Switch Event Port Selection 0 | 0x3FF00 | Port 2 Master/Slave | [21], 0x--N----- | 0 | Master or slave for port 2, 1=master, 0=slave |
| Stream Switch Event Port Selection 0 | 0x3FF00 | Port 2 ID | [20:16], 0x--NN---- | 0 | Select port ID for port 2 event gen |
| Stream Switch Event Port Selection 0 | 0x3FF00 | Port 1 Master/Slave | [13], 0x----N--- | 0 | Master or slave for port 1, 1=master, 0=slave |
| Stream Switch Event Port Selection 0 | 0x3FF00 | Port 1 ID | [12:8], 0x----NN-- | 0 | Select port ID for port 1 event gen |
| Stream Switch Event Port Selection 0 | 0x3FF00 | Port 0 Master/Slave | [5], 0x------N- | 0 | Master or slave for port 0, 1=master, 0=slave |
| Stream Switch Event Port Selection 0 | 0x3FF00 | Port 0 ID | [4:0], 0x------NN | 0 | Select port ID for port 0 event gen |

This info is also found online in [AM025](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/) for [Stream Switch Event Port Selection 0](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/Stream_Switch_Event_Port_Selection_0-CORE_MODULE-Register) and [Stream Switch Event Port Selection 1](https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/Stream_Switch_Event_Port_Selection_1-CORE_MODULE-Register).


<u>Example Port Selection 0</u>

in C/C++
```c++
// Stream_Switch_Event_Port_Selection_0
// This is necessary to capture the Port_Running_0 and Port_Running_1 events
// Port 0 - Master/ID=1, Port 1 - Slave/ID=1
aiex.npu.write32 { column = 0 : i32, row = 4 : i32, address = 0x3FF00 : ui32, value = 0x121 : ui32 }
aiex.npu.write32 { column = 0 : i32, row = 4 : i32, address = 0x3FF04 : ui32, value = 0x0 : ui32 }
```
in Python
```python
# def master(port):
#     return port | (1 << 5)

# def slave(port):
#     return port

npu_write32(column=0, row=4, address=0x3FF00, value=pack4bytes(0, 0, slave(1), master(1)),)  # port 1 is FIFO0?
npu_write32(column=0, row=4, address=0x3FF04, value=pack4bytes(0, 0, 0, 0),)
```

### Configure shimDMA

The shimDMA needs to be configured to write the trace stream data to a valid location in DDR memory to be read by the host code. In the case of the NPU, we can use a template like the following where the main parameters that need to be defined include `buffer_length`, `buffer_offset`, `bd_id`, `ddr_id`, and `column`.

* `buffer_length` - is the expected trace buffer size in bytes and should match what's expected from the host code when reading the trace buffer.
* `buffer_offset` - specifies in bytes where the trace buffer starts in the output buffer and occurs after the main output buffer ends. If the output buffer size in words is 65,536, then the buffer offset would be 4*65,536 = 262,144 bytes.
* `bd_id` - unique bd (out of 16 bds) to program for data movement. Since we're delcaring this manually, it's important that we dont' overlap with existing (and possibly auto-declared bd id values). In the matmul design, we needed this value to be at least 13.
* `column` - this shimDMA's column
* `ddr_id` - very important to indicate which inout buffer DDR region we're mapping to. 

An example ddr_id to inout buffer mapping is below:
| ddr ID value | buffer |
|-----------|--------|
| 0 | inout0 |
| 1 | inout1 |
| 2 | inout2 |

<u>Example shimDMA config</u>

in C/C++
```c++
aiex.npu.writebd { bd_id = 3 : i32,
                   buffer_length = 16384 : i32,
                   buffer_offset = 262144 : i32,
                   enable_packet = 0 : i32,
                   out_of_order_id = 0 : i32,
                   packet_id = 0 : i32,
                   packet_type = 0 : i32,
                   column = 0 : i32,
                   d0_stepsize = 0 : i32,
                   d0_size = 0 : i32,
                   d0_stride = 0 : i32,
                   d0_wrap = 0 : i32,
                   d1_stepsize = 0 : i32,
                   d1_wrap = 0 : i32,
                   d1_size = 0 : i32,
                   d1_stride = 0 : i32,
                   d2_stepsize = 0 : i32,
                   d2_size = 0 : i32,
                   d2_stride = 0 : i32,
                   iteration_current = 0 : i32,
                   iteration_stepsize = 0 : i32,
                   iteration_wrap = 0 : i32,
                   iteration_size = 0 : i32,
                   iteration_stride = 0 : i32,
                   lock_acq_enable = 0 : i32,
                   lock_acq_id = 0 : i32,
                   lock_acq_val = 0 : i32,
                   lock_rel_id = 0 : i32,
                   lock_rel_val = 0 : i32,
                   next_bd = 0 : i32,
                   row = 0 : i32,
                   use_next_bd = 0 : i32,
                   valid_bd = 1 : i32}
// Set start BD to our shim bd_Id (3)
aiex.npu.write32 { column = 0 : i32, row = 0 : i32, address = 0x1D20C : ui32, value = 0x3 : ui32 }
```
in Python
```python
npu_writebd(
    bd_id=3,
    buffer_length=16384,
    buffer_offset=262144,
    enable_packet=0,
    out_of_order_id=0,
    packet_id=0,
    packet_type=0,
    column=0,
    column_num=1,
    d0_size=0,
    d0_stride=0,
    d1_size=0,
    d1_stride=0,
    d2_stride=0,
    ddr_id=2,
    iteration_current=0,
    iteration_size=0,
    iteration_stride=0,
    lock_acq_enable=0,
    lock_acq_id=0,
    lock_acq_val=0,
    lock_rel_id=0,
    lock_rel_val=0,
    next_bd=0,
    use_next_bd=0,
    valid_bd=1,
)
```    

## XRT utilites ([xrt.py](./xrt.py))
XRT wrapped utilities

* class `AIE_Applications`
* class `AIE_Buffer`
* class `AIE_Application_Error`
* `read_insts`
* `setup_aie`
* `extract_trace`
* `write_out_trace`
* `execute`

## Machine Language (ML) utilites ([ml.py](./ml.py))
ML related utilties

* class `CSVLogger`
* `load_class_label`
* `unpickle`
* `fuse_single_conv_bn_pair`
* class `DataShaper`
