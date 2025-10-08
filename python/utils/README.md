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
improt aie.utils.xrt as xrt_utils
```
Thereafter, functions defined in the particular utils file such as `trace_utils` can be called via `trace_utils.configure_packet_tracing_aie2(...)`.

- [Test utilities](#test-utilites-testpy) ([test.py](./test.py))
- [Trace utilities](#trace-utilites-tracepy) ([trace.py](./trace.py))
    - [Trace Mechanisms and Explanations](#trace-mechanisms-and-explanations)
    - [Trace parser](#trace-parser-parse_tracepy) ([parse_trace.py](./parse_trace.py))
    - [Trace parser - eventIR based](#trace-parser---eventir-based-parse_eventirpy) ([parse_eventIR.py](./parse_eventIR.py))
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

Trace utilities are designed to take the low level tile cofigurations need to configure trace and wrap them into convenient wrapper functions. We will go over descriptions of the functions and then dive deeper into Trace mechanisms and explanations.

* `class GenericEvent`
* `PortEventCodes`, `MemTileEventcodes`, `ShimTileEventCodes`
    * event codes for port events for the different tile types: core, memtile, shimtile
* `class PacketType` 
    * We use the packet type field in the packet header to help differentiate the tile that the packet came from. Since packet types don't inherently have meaning, we assign numerical values to each tile type: core, mem (for core), shimtilem, memtile
* `class PortEvent`, `class MemTilePortEvent`, `class ShimTilePortEvent`
    * class for port events to define accesses and `get_register_writes`
* `isShimTile`, `isMemTile`
    * Placeholder functions to test if a particular tile is a shim or mem tile. The current definitions is appropriate for phoenix and strix devices but these functions should be expanded to account for varying devcie types based on device model
* `pack4bytes`
    * Pack 4 bytes into a 32-bit word

* `configure_coretile_tracing_aie2`, `configure_memtile_tracing_aie2`, `configure_shimtile_tracing_aie2`
    * This function configures the a tile's trace unit given a set of configurations as described below:
        function arguments:
        * `tile` - ocre tile to configure
        * `start` - start event. We generally use a global broadcast signal to synchronize the start event for multiple cores.
        * `stop` - stop event. We generally use a global broadcast signal to synchronize the stop event for multiple cores.
        * `events` - array of 8 events to trace
        * `enable_packet` - enables putting event data into packets
        * `packet_id` - packet id or flow id used to route the packets through the stream switch
        * `packet_type` - packet type is an arbitrary field but we use it to describe the tile type the packets are coming from
        * `shim_burst_length` - burst size for shim dma. Default is 256B but can be as low as 64B which may be helpful for small trace designs

* `configure_timer_ctrl_coretile_aie2`, `configure_timer_ctrl_memtile_aie2`,`configure_timer_ctrl_shimtile_aie`
    * Configures timer in each tile type to reset based on an `event`

* `configure_broadcast_core_aie2`
    *  Configure broadcast event based on an internal triggered event. 
        function arguments:
        * `num` - broadcaast number we want to broadcast on 
        * `event` - the triggering broadcast event

* `configure_event_gen_core_aie2`
    * Generate an `event` at the given `tile`. This event can broadcasted and use by all tiles in the device to synchronize to.

* `configure_shimtile_dma_tracing_aie2`
    * Configure shim tile's DMA for tracing. This configures the shim tile / bd to process a specficic `packet id` and `packet type`. It also configures the address patch. Note that we can call this multiple times for each `packet id`/ `packet type` but mapped to the same `ddr_id`, `size`, and `offset` and the packets will be written to the output location as they come in for all `packet id`/ `packet type` listed

* `configure_coretile_packet_tracing_aie2`, `configure_memtile_packet_tracing_aie2 `, `configure_shimtile_packet_tracing_aie2`
    * Wrapper to configure the core tile and shim tile for packet tracing. This does the following:
        1. Configure core tile based on start/ stop, events, and flow id. The flow id needs to be unique per flow.
        2. Configure timer based on broadcast event (default is 15). This ensures all tiles keying off this event has a synchronized timer so their trace are synchronized. This event is also used as the start event for tracing.
        3. Configure shim tile to receive this flow and move the data to offset/ size.
    It does this by calling `configure_coretile_tracing_aie2`, `configure_time_ctrl_coretile_aie2` and `configure_shimtile_dma_tracing_aie2`. 

* `configure_packet_tracing_flow`
    * Wrapper around packeflows to itereate over tiles_to_trace and route them to the shim for outputing the trace to L3 memory. This uses default values for the packet id that increases for each tile we trace, starting with 1. This should match the tile trace config that's set by configure_coretile_packet_tracing_aie2. 
    * *NOTE* - Because we do it this way, we inherently cannot trace more than 31 tiles.

        Function arguments:
        * `tiles to trace` - array of tiles to trace
        * `shim tile` - Single shim tile to configure for writing trace packets to DDR

        An example use case would be:
        ```python
        trace_utils.configure_packet_tracing_flows(tile_to_trace, ShimTile)
        ```

* `configure_shim_trace_start_aie2`
    * Configure the shim tile to support packet tracing via:
        1. Set an event generation to create a custom user event 1 (127, 0x7f)
        2. Custom event also triggers a broadcast event (by default broadcast 15)
        3. Custom event also resets timer (will be true for all tiles) so all timers are synchronized
        The actual shim dma config is done via configure_shimtile_tracing_aie2 but this tends to be done for each tile we're tracing.

        Function arguments:
        * `brdcst_num` - which broadcast number to use (1-15)
        * `user_event` - Which user event do we want to generate which will be used to reset local timers and be broadcasted out on the `broadcast_num`

* `gen_trace_done_aie2`
    * Generate a done event (broadcasted shim user event) that the other tile will use as stop event

* `configure_packet_tracing_aie2` (packet switched multi-tile tracing)
    * This wrapper function iterates over the `tiles_to_trace` array and calls the right version of `configure_*tile_packet_tracing_aie2`. A key distinction is made to choose the right start and stop event depending on the tile type. We pass in 3 sets of optional event arguments that allows them to be customized depending on the tile type.
        
        Function arguments:
        * `tiles to trace` - array of tiles to trace
        * `shim tile` - Single shim tile to configure for writing trace packets to DDR
        * `size` - trace buffer size (in bytes)
        * `offset` - offest (in bytes) where trace buffer data should begin. By default, this is 0 but can be >0 if we share a buffer with an output.
        * `enable_token` - enable token generation for shimdma. Not recommended since we generally have our dma size > trace data size which we don't always know how big it needs to be.
        * `ddr_id` - which XRT buffer to use where 0 -> group_id(3) ... 4 -> group_id(7). We generally put trace last so we use ddr_id=4.
        * `start_user_event` - which user event do we use as a start event
        * `stop_user_event` - which user event do we use as a stop event
        * `start_broadcast_num` - which broadcast number do we send the start user event
        * `stop_broadcast_num` - which broadcast number do we send the stop user event
        * `coretile_events` - which 8 events do we use for all coretiles in array
        * `memtile_events` - which 8 events do we use for all memtiles in array
        * `shimtile_events` - which 8 events do we use for all shimtiles in array
        * `shim_burst_length` - burst size for shim dma. Default is 256B but can be as low as 64B which may be helpful for small trace designs

        An example use case would be:
        ```python
        trace_utils.configure_packet_tracing_aie2(tile_to_trace, ShimTile, opts.trace_size)
        ```

* `configure_simple_tracing_aie2` (**DEPRECATED** cicuit switched single tile tracing)
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

## Trace Mechanisms and Explanations

The basic concept for trace configuration as summarized in [section-4b](../../programming_guide/section-4/section-4b/). MOre details about the trace hardware can be found for AIE-ML/AIE2 at [am020](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Trace).

### Trace Packet Routing
Digging one level lower, tracing can be configured such that trace data is moved via circuit switch routing or packet switched routing. The deprecated `configure_simple_tracing_aie2` uses circuit switch tracing but this mechanism utilizes a dedicate stream along the stream switch path and limits the number of parallel tiles that can be traced. The preferred default mechanism is to use packet swtiched routing instead. This has the benefit of using a shared stream to route multiple tiles' trace packets. In practice, if a large amount trace data is being produced among a large number of tiles and aggregated into a single stream, there can be a limit to how much data that stream can support which may exert back pressure can cause overrun of trace data leading to invalid trace results. One limitation of packet switched routing is the additional packet header prepended to each packet (32b header for 7x 32b of data payload). This reduces the effective bandwidth of the trace data but the benefit of packet switched routing far outweigh this overhead limitation. 

### Trace Array Level Configuration (packet switched routing)
We have already discussed configuring individual trace units in each tile to enable tracing and packetization of the trace data, configuring packet flows to route the trace data packets to a shim, and configuring the shim to write that data to DDR. However, a key aspect of full array level configuration involves supporting multi-tile trace which requires synchronization of trace data. This is done via using broadcasted user events as both local timer reset and start and stop synchronization, as explained below:

1. Configure shim to generate a custom user event (#1) and broadcast event (#15) throughout array
2. Reset all timers in shim and traced tiles based on this broadcast event so all timers are synchronized. NOTE: In practice, there is a slight delay since the delay of this signal can be a few clock cycles between tiles. 
3. Configure tiles to use this broadcast event (#15) as the start event
4. Continue with rest of runtime sequence (e.g. data movement for input and output buffers)
5. Generate another user event (#0) and broadcast event (#14) throughout array. This will be used as the trace stop event for all tiles

### Available Events for Tracing - `trace_events_enum.py`

`trace_events_enum.py` contains a list of all traceable events on AIE-ML devices.
These include events on compute cores (`CoreEvent`), memory modules (`MemEvent`), shim tiles (`ShimEvent`) and mem tiles (`MemTileEvent`).
When specifying a list of events to `configure_packet_tracing_aie2` and other functions, you can refer to events either by their name or their numeric values:
```
configure_packet_tracing_aie2(..., events=[0x4B, 0x22, 0x21, 0x25])
# or, equivalently:
configure_packet_tracing_aie2(..., events=[CoreEvent.INSTR_EVENT_1, CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_VECTOR, CoreEvent.INSTR_LOCK_RELEASE_REQ])
```

#### Port Events

There is a set of events that fire on certain activity on data memory ports.
These are `PORT_IDLE_0` through `PORT_IDLE_7`, `PORT_RUNNING_0` through `PORT_RUNNING_7`, `PORT_STALLED_0` throught `PORT_STALLED_7` and finally `PORT_TLAST_0` through `PORT_TLAST_7`.
You have to specify on which port the tracing engine should listen for each those events.
In hardware, this is done by configuring registers `0x3FF00` and `0x3FF04` in the core tile (this is different in memtile and shimtile).
The Python tracing utilities abstract this in `configure_packet_tracing_aie2`; you only have to specify the event as a `PortEvent` along with the corresponding port as follows:

```
configure_packet_tracing_aie2(
    ...,
    events=[
        PortEvent(CoreEvent.PORT_RUNNING_0, 1, master=True)
        # This will emit an event whenever master port 1 is running.
    ]
)
```

`PortEvent` is defined in [trace.py](../../python/utils/trace.py) and `CoreEvent` is defined in [trace_events_enum.py](../../python/utils/trace_events_enum.py). Likewise for memtiles and shimtiles, we have `MemTilePortEvent` and `ShimTilePortEvent` in [trace.py](../../python/utils/trace.py) and `MemTileEvent` and `ShimTileEvent` are in [trace_events_enum.py](../../python/utils/trace_events_enum.py).

### Configure tile trace settings
Under the hood of `configure_coretile_tracing_aie2`/ `configure_memtile_tracing_aie2`/ `configure_shimtile_tracing_aie2`, we perform trace configurations by writing specific values to trace configuration registers. This is done within the `aiex.runtime_sequence` block, where we call a set of configuration register writes (`aiex.npu.write32`) to configure the tile trace units and (`aiex.npu.writebd`) to configure the shimDMA. 

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

**Note** that Trace Control 0 is all you need for circuit switched flows. For packet switched flows, however, you will also need to configure Trace Control 1. Here, the packet type matches the tile types we define to support post-run parsing. Here is a table of trace unit types and packet type.

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


There is an extensive lists of trace events in the [trace_events_enum.py](../../python/utils/trace_events_enum.py), but we will only list a few common ones here.
| Some common events | Associated CoreEvent name | hex value | dec value |
|--------------------|---------------------------|----------|-----------|
| True                       |TRUE| 0x01| 1 |
| Stream stalls              |STREAM_STALL| 0x18| 24 |
| Core Instruction - Event 0  |INSTR_EVENT_0| 0x21| 33|
| Core Instruction - Event 1  |INSTR_EVENT_1| 0x22| 34 |
| Vector Instructions (e.g. VMAC, VADD, VCMP) |INSTR_VECTOR| 0x25|  37 |
| Lock acquire requests      |INSTR_LOCK_ACQUIRE_REQ| 0x2C|  44 |
| Lock release requests      |INSTR_LOCK_RELEASE_REQ| 0x2D|  45 | 
| Lock stall                 |LOCK_STALL| 0x1A|  26 |
| Core Port Running 1        |PORT_RUNNING_1| 0x4F|  79 |
| Core Port Running 0        |PORT_RUNNING_0| 0x4B|  75 | 

**NOTE**: The "Core Instruction - Event 0/1" are special intrinsics you can add to your kernel code to trigger an event during the running of your core program. Within the kernel code, they look like:
```c++
event0();
<my kernel code>
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
| ddr ID value | buffer | group_id |
|-----------|--------|----------|
| 0 | inout0 | 3 |
| 1 | inout1 | 4 |
| 2 | inout2 | 5 |
| 3 | inout3 | 6 |
| 4 | inout4 | 7 |

By default, trace in our wrapper functions will map to inout4 (group_id 7).

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
    ddr_id=4,
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

### <u>Trace parser ([parse_trace.py](./parse_trace.py))</u>
The text file generated by the host code (`test.cpp` or `test.py`) are formatted as 32-bit hex values, one per line. This python script parses the raw trace packet data and creates a waveform json file for view on Perfetto http://ui.perfetto.dev. The script syntax is:

```bash
parse_trace.py --input trace.txt --mlir build/aie_trace.mlir --output parse_eventIR_vs.json
```

* **--filename** : Input trace packet text file. This is generated during the running of our python host code
* **--mlir**     : MLIR source. This is needed to parse what events and tiles we are monitoring to generate labels for our waveform visualizer.
* **--colshift (optional)** : runtime column shift. This specifies how much the actual design was shifted from the default position when it was scheduled and called. The reason we need this is becuase even if our design is configured for column 0, the actual loading and execution of the design may place it in column 1, 2, 3 etc. We account for this shift since the parser needs to match the actual column location of the generated trace data. For npu devices (phoenix), this is typically 1 while npu2 (strix) uses 0. The script should be able to automatically figure out the starting column and set this correctly but can be overrided via this argument.

    **NOTE** - the underlying tools currently default to column 1 to avoid using column 0 on Ryzen AI since that column does not have a shimDMA and is therefore avoided at the moment. 
* **--output** : output json file


### <u>Trace parser - eventIR based ([parse_eventIR.py](./parse_eventIR.py))</u>
The text file generated by the host code (`test.cpp` or `test.py`) are formatted as 32-bit hex values, one per line. This python script executes a number of steps in order to transform it from trace packet text file into a waveform json file.

**NOTE** - There seems to be some inconsistencies in the results generated by this parser. As of now, it is used to compare to existing the `hwfrontend` tool only.

The script syntax is:

```bash
parse_eventIR.py --input trace.txt --mlir build/aie_trace.mlir --output parse_eventIR_vs.json
```
* **--input** : Input trace packet text file. This is generated during the running of our python host code
* **--mlir**     : MLIR source. This is needed to parse what events and tiles we are monitoring to generate labels for our waveform visualizer.
* **--colshift** : runtime column shift. This specifies how much the actual design was shifted from the default position when it was scheduled and called. The reason we need this is becuase even if our design is configured for column 0, the actual loading and execution of the design may place it in column 1, 2, 3 etc. We account for this shift since the parser needs to match the actual column location of the generated trace data. Usually 1 is the right value. **NOTE** - the underlying tools currently default to column 1 to avoid using column 0 on Ryzen AI since that column does not have a shimDMA and is therefore avoided at the moment.

The parse script create a temporary directory `tmpTrace` performs the following steps within that folder:
1. [Fixes raw trace data](#1-fixes-raw-trace-data)
1. [Parse MLIR to build event table](#2-parse-mlir-to-build-event-table)
1. [Create .target file](#3-create-target-file)
1. [Create config.json](#4-create-configjson)
1. [Run Vitis/aietools hwfrontend utility to parse raw trace data --> generates eventIR.txt](#5-run-vitisaietools-hwfrontend-utility-to-parse-raw-trace-data----generates-eventirtxt)
1. [Convert eventIR.txt to perfetto_compatible.json](#6-convert-eventirtxt-to-perfetto_compatiblejson)
* [Additional Tips](#tips)

#### <u>1. Fixes raw trace data</u>
We prepend `0x` before each hex line and save it `prep.<trace file>` since the `hwfrontend` utility expects it.

#### <u>2. Parse MLIR to build event table</u>
The MLIR parser is pretty rudimentary as it scans the source mlir file looking for `aiex.npu.write32` calls and does a pattern match for trace unit config address and then grab the hex events, which it looks up from an internal table to provide waveform labels. It would be better to use an MLIR pass that already has the config information and cross reference it with a more official event-to-label lookup table instead. 

#### <u>3. Create .target file</u>
Create a dummy file (`.target`) in the `tmpTrace` with the file content 'hw' since `hwfrontend` utility expects it.

#### <u>4. Create config.json</u>
This step uses the information from the MLIR parse step to create a fixed config that has the matching events in a `config.json` file. This file is used by `hwfrontend` when it's parsing the trace packet data. While core tile config seems correct, memtile and shimtile are not yet supported or tested.

#### <u>5. Run Vitis/aietools hwfrontend utility to parse raw trace data --> generates eventIR.txt</u>
This is the main parse utility that generates a much more friendly eventIR file format. This utilty is the same one used by the adf tools for aiesimulator. However, the utility is very particular and some combinations of trace packet data might confuse the parser or cause an error. See the **Tips** section at the end for workarounds to known issues.

#### <u>6. Convert eventIR.txt to perfetto_compatible.json</u> 
While the Perfetto-compliant json file format is not the same as the eventIR file format. The conversion between them is more straightforward that between trace packets and Perfetto-compliant json. Having said that, it is still possible this pass to be further tested and improved.

#### <u>Tips</u>
<u>Workaround 1</u>: For case where the start event is 1 or maybe in general, the trace output might have a few packets with just `0xdbffdbff` data. These seem to give the following error and needs to have those packets deleted up to an actual valid event packet.
```bash
CRITICAL WARNING: [hwanalyze 77-5570] trace_out_hex3:1 Start Frame for Tile(2, 4) Module: cm looks to be missing as trace configuration is not available.
```
<u>Workaround 2</u>: If the start timer value is too large, it reports an error:
```bash
WARNING: [hwanalyze 77-5569] trace_out_hex2:1 Overrun Tile(2, 4) Module: cm. Hence further decoding of this particular module will be skipped.
```
So reducing the start frame from something like:
```bash
0xf4000000
0x00a93c62
```
to
```bash
0xf0000000
0x0005d0f7
```
which reduces the timer from 11,091,042 cycles to 381,175 seems to fix it.

## XRT utilites ([xrt.py](./xrt.py))
XRT wrapped utilities. These classes and utilities help simplify the the declaration and instantiation of XRT components in the host code.

In particular, `setup_and_run_aie` is a helpful convenience wrapper to simplify the setup and runnin of kernel with 1 or 2 inputs buffers and 1 output buffer. See [vector_scalar_mul](../../programming_examples/basic/vector_scalar_mul/) for an template example of how this is ued.

* class `AIE_Application`
    * This class configures and invokes the XRT components needed to run an AIE Application. This includes xrt.device, xrt.kernel, xrt.hw_context and XRT buffers as enacpuslated by the AIE_Buffer class. You can use this class to simplify and reduce the amount of code needed to set up an AIE application.
        * `__init__` - Registers xclbin to set up the device, hw context and kernel. This also sets up the instruction stream
        * `register_buffer` - Registers an AIE_Buffer class object given group_id, datatype and shape
        * `run` - This syncs the instruction buffer to the device and then invokes the `call` function before wait for the call to complete
        * `call` - Wrapper for xrt.kernel function passing in opcode and buffers objects
* class `AIE_Buffer`
    * This class wraps up access to the xrt.bo buffer object where sync calls are added to read and write calls to ensure data is synchronized.
    * `__init__` - Declare xrt.bo object given group_id, datatype, shape
    * `read` - Synchronize data from device before reading xrt.bo data
    * `write` - Write data to xrt.bo and synchronize data to device
    * `sync_to_device` - Wrapper for xrt.bo.sync call (to device)
    * `sync_from_device` - Wrapper for xrt.bo.sync call (from device)
* class `AIE_Application_Error`
* `read_insts` - Read instruction stream from text file and reformat it to be passed into the instructoin buffer for the xrt.kernel call
* `setup_aie`
    * Sets up the AIE application with support for up to 2 input buffers, 1 output buffer, and an optional trace buffer. Under the hood, we call declare an AIE_Application object and register the buffers used given the buffer datatype and shapes. 
* `execute`
    * Wrapper function to write buffer arguments into registered input buffers, then call `run` function for AIE Application, and finally return the output buffer data.
* `extract_trace`
    * Wrapper function to separate output data and trace data from a single output buffer stream
* `write_out_trace`
    * Wrapper function to write trace buffer values to a text file
* `setup_and_run_aie`
    * This wrapper function abstracts the full set of functions to setup the aie and run the kernel program including check for functional correctness and reporting the run time. Under the hood, we call `setup_aie` to set up the AIE application before calling `execute` and checking results. The datatypes and shape for the 2 inputs and 1 output buffers are passed in as arguments, along with the gold reference data to compare it against. Trace buffers is also written out to a text file if trace is enabled. 

## Machine Language (ML) utilites ([ml.py](./ml.py))
ML related utilties

* class `CSVLogger`
* `load_class_label`
* `unpickle`
* `fuse_single_conv_bn_pair`
* class `DataShaper`
