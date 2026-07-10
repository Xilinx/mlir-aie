# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Trace utilities

These utilities configure AIE hardware tracing and parse the resulting trace
data. Tracing is expressed declaratively: you describe *what* to trace with
`aie.trace` ops, and trace lowering (`aie-insert-trace-flows`) turns that into
the register writes, packet flows, and shim DMA configuration that implement it.

## Public API

* `configure_trace(tiles_to_trace, ...)`
    * Emit `aie.trace` ops for a list of tiles. Each op declares the start/stop
      events, the up-to-8 events to monitor, and (via auto-detection) the tile
      type. Call this once with all the tiles you want to trace. Per-tile-type
      event lists can be overridden with `coretile_events`, `coremem_events`,
      `memtile_events`, `shimtile_events`; otherwise sensible defaults are used.
      Port activity is specified with `PortEvent` (see Port Events below).

* `start_trace(trace_size, reuse_output_buffer=False, ...)`
    * Emit, inside an `aie.runtime_sequence`, the `aie.trace.host_config` +
      `aie.trace.start_config` ops that activate the configured traces and
      describe the host trace buffer. `trace_size` is the buffer size in bytes.
      By default trace lowering appends a dedicated trace-buffer argument to the
      runtime sequence (it lands at the tail, so enabling trace never shifts the
      data arguments' indices). Set `reuse_output_buffer=True` to instead write
      trace data into the tail of the last output buffer, saving a host buffer.

* `TraceConfig`
    * Host-side trace configuration (buffer size, output file,
      `reuse_output_buffer`, control-packet options). Carried by `NPUKernel`
      and used by the runtime to size/append the trace buffer and to read the
      trace data back after a run.

* `parse_trace(trace_buffer, mlir_module_str)`
    * Parse a raw trace buffer into Perfetto-compatible events, using the lowered
      MLIR to recover which tiles/events were traced (see Trace parser below).

* `configure_packet_ctrl_flow`, `config_ctrl_pkts_aie`
    * Lower-level helpers for control-packet-based trace configuration.

* Helpers in `utils.py` (`pack4bytes`, `create_ctrl_pkt`, `get_cycles`,
  `get_cycles_summary`, `print_cycles_summary`, `get_vector_time`,
  `split_trace_segments`, ...) are re-exported and documented in source.

The high-level IRON entry point is `Runtime.enable_trace(...)`, which builds a
`TraceConfig` and calls `configure_trace` / `start_trace` for you.

## The host trace buffer is a runtime_sequence operand

Every host buffer a kernel uses -- data, trace, control-packet -- is an argument
of the `aie.runtime_sequence`, and argument *i* maps to XRT host BO *i* (after
the fixed opcode/instr/ninstr scalars). Trace lowering makes the trace buffer one
of those arguments by appending it, so:

* the trace buffer's index is wherever the tail is, not a fixed slot, and
* the host BO count is simply the (lowered) runtime_sequence argument count.

Both the compiler (when emitting the trace buffer's address patch) and the host
(when allocating and passing the trace buffer) agree on this index by
construction; neither hardcodes it.

## Trace Mechanisms and Explanations

The basic concept for trace configuration is summarized in [section-4b](../../programming_guide/section-4/section-4b/). More details about the trace hardware can be found for AIE-ML/AIE2 at [am020](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Trace).

### Trace Packet Routing
Trace data is moved via packet-switched routing: tiles' trace streams are packetized and share a stream to a shim, which writes them to DDR. Packet switching lets multiple tiles' trace packets share a single stream. The trade-off is an extra packet header (32b header per 7x 32b of data payload), which reduces effective trace bandwidth; in practice the flexibility outweighs the overhead. If a large amount of trace data from many tiles is aggregated into one stream, the stream can back-pressure and overrun, producing invalid trace results -- in that case trace fewer tiles per stream or reduce the event rate.

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
When specifying a list of events to `configure_trace` (via the per-tile-type event lists), you can refer to events either by their name or their numeric values:
```
configure_trace(..., coretile_events=[0x4B, 0x22, 0x21, 0x25])
# or, equivalently:
configure_trace(..., coretile_events=[CoreEvent.INSTR_EVENT_1, CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_VECTOR, CoreEvent.INSTR_LOCK_RELEASE_REQ])
```

#### Port Events

There is a set of events that fire on certain activity on data memory ports.
These are `PORT_IDLE_0` through `PORT_IDLE_7`, `PORT_RUNNING_0` through `PORT_RUNNING_7`, `PORT_STALLED_0` throught `PORT_STALLED_7` and finally `PORT_TLAST_0` through `PORT_TLAST_7`.
You have to specify on which port the tracing engine should listen for each those events.
In hardware, this is done by configuring registers `0x3FF00` and `0x3FF04` in the core tile (this is different in memtile and shimtile).
The Python tracing utilities abstract this in `configure_trace`; you only have to specify the event as a `PortEvent` along with the corresponding port as follows:

```python
configure_trace(
    tiles_to_trace,
    coretile_events=[
        PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True)
        # This will emit an event whenever DMA channel 0 input is running.
    ]
)
```

`PortEvent` takes the following arguments:
- `code`: The PORT_RUNNING_N event (determines which monitor slot 0-7 to use)
- `port`: The port bundle type (WireBundle.DMA, WireBundle.North, etc.)
- `channel`: Channel number within the bundle
- `master`: True for input to tile (S2MM), False for output from tile (MM2S)

`PortEvent` is defined in [events/__init__.py](events/__init__.py) and `CoreEvent` is defined in [events/aie2.py](events/aie2.py) (generated during build). Likewise for memtiles and shimtiles, we have `MemTilePortEvent` and `ShimTilePortEvent`.

### Configure tile trace settings
Under the hood, trace lowering performs trace configuration by writing specific values to trace configuration registers. This is done within the `aiex.runtime_sequence` block, where a set of configuration register writes (`aiex.npu.write32`) configure the tile trace units and (`aiex.npu.writebd`) configures the shimDMA.

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
* `buffer_offset` - specifies in bytes where the trace buffer starts in the output buffer. It is 0 for a dedicated trace buffer; when reusing an output buffer it occurs after the main output data ends (e.g. if the output buffer is 65,536 words, the offset is 4*65,536 = 262,144 bytes).
* `bd_id` - unique bd (out of 16 bds) to program for data movement. Since we're declaring this manually, it's important that we don't overlap with existing (and possibly auto-declared) bd id values. In the matmul design, we needed this value to be at least 13.
* `column` - this shimDMA's column
* `ddr_id` - the host argument index of the trace buffer (the runtime_sequence operand it lands on). This is the value the shim DMA's address patch is keyed to; the host must pass the trace buffer at the same index.

Because trace lowering appends the trace buffer as the last runtime_sequence argument, its `ddr_id` is the index of that appended operand (or, with `reuse_output_buffer`, the index of the reused output operand). The host argument index N maps to XRT group_id N+3 (the first three args are the opcode/instr/ninstr scalars).

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

### <u>Trace parser ([parse_trace.py](./parse.py))</u>
The text file generated by the host code (`test.cpp` or `test.py`) are formatted as 32-bit hex values, one per line. This python script parses the raw trace packet data and creates a waveform json file for view on Perfetto http://ui.perfetto.dev. The script syntax is:

```bash
parse_trace.py --input trace.txt --mlir build/aie_trace.mlir --output parse_eventIR_vs.json
```

* **--filename** : Input trace packet text file. This is generated during the running of our python host code
* **--mlir**     : MLIR source. This is needed to parse what events and tiles we are monitoring to generate labels for our waveform visualizer.
* **--colshift (optional)** : runtime column shift. This specifies how much the actual design was shifted from the default position when it was scheduled and called. The reason we need this is becuase even if our design is configured for column 0, the actual loading and execution of the design may place it in column 1, 2, 3 etc. We account for this shift since the parser needs to match the actual column location of the generated trace data. For npu devices (phoenix), this is typically 1 while npu2 (strix) uses 0. The script should be able to automatically figure out the starting column and set this correctly but can be overrided via this argument.

    **NOTE** - the underlying tools currently default to column 1 to avoid using column 0 on Ryzen AI since that column does not have a shimDMA and is therefore avoided at the moment.
* **--output** : output json file


### <u>Trace parser - eventIR based ([event_ir.py](./event_ir.py))</u>
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
"""

from .config import TraceConfig
from .parse import parse_trace
from .setup import (
    configure_packet_ctrl_flow,
    config_ctrl_pkts_aie,
    configure_trace,
    start_trace,
)
from .utils import (
    parity,
    extract_tile,
    pack4bytes,
    create_ctrl_pkt,
    get_kernel_code,
    extract_buffers,
    get_cycles,
    get_cycles_summary,
    print_cycles_summary,
    get_vector_time,
    split_trace_segments,
)
