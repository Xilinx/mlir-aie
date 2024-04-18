<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Python Utilities</ins>

The python utilties are designed to simplify commonly repeated tasks and wrap them up into helper functions. They are divided into separate categories and can be added to any python code, for example, via:
```
import aie.utils.trace as trace_utils
```
Thereafter, fucntions defined in the file can be called via `trace_utils.configure_simple_tracing_aie2(...)`.

- [XRT utilities (xrt.py)](./xrt.py)
- [Test utilities (test.py)](./test.py)
- [Trace utilities (trace.py)](./trace.py)
- [Machine Learning(ML) utilities (ml.py)](./ml.py)

## <u>Trace utilites (trace.py)</u>

configure_simple_tracing_aie2


Within the `func.func @sequence` block, we add a set of configuration register writes (`aiex.ipu.write32`) to configure the tile trace units and the shimDMA. 



### <u>Configure tile trace settings</u>
For a give AIE2 tile, we configure the trace control for the tile core and tile memory separately. There are 4 registers we generally use to configure the trace behavior. 2 are for configuring the general trace control and the other 2 are to specify which events our tile's trace hardware is monitoring.

The table below describes the general trace control registers.

| Config Register | Address | Field | Bits | Reset Value | Description |
|-----------------|---------|-------|------|-------------|-------------|
| Trace Control 0 | 0x340D0 | Stop Event | [30:24], 0xNN------ | 0 | Event to stop trace capture | 
| Trace Control 0 | 0x340D0 | Start Event | [22:16], 0x--NN---- | 0 | Event to start trace capture |
| Trace Control 0 | 0x340D0 | Mode | [1:0], 0x-------N | 0 | Trace mode. 00=event-time, 01=event-PC, 10=execution |
| Trace Control 1 | 0x340D4 | Packet Type | [14:12], 0x----N--- | 0 | Detination trace packet - packet type |
| Trace Control 1 | 0x340D4 | Packet ID | [4:0], 0x------NN | 0 | Detination trace packet - packet ID |

Note that Trace Control 0 is all you need for circuit switched flows. For packet switched flows, however, you will also need to configure Trace Control 1. Here, the packet type matches the types we defined above (generally, 0 for core and 1 for memory). The ID is arbitrary but must match the packet flows you define or else the packets will not get routed.

<u>Example</u>
```
// Start event = 1, Stop event = 0, Mode = event-time
aiex.ipu.write32 { column = 0 : i32, row = 4 : i32, address = 0x340D0 : ui32, value = 0x10000 : ui32 }
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

There is an extensive lists of trace events but here, we will only describe a few key ones.
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

<u>NOTE</u>: The "Core Instruction - Event 0/1" are special intrinsics you can add to your kernel code to trigger an event during the running of your core program. Within the kernel code, they look like:
```
event0();
...
event1();
```
This can be placed at the beginning and end of your code block to estimate the total execution time of your kernel program.

<u>Example</u>
```
// Events 0-3 monitored
// ------------------------
// Vector instrucitons (0x25)
// Core Instruction - Event 0 (0x21)
// Core Instruction - Event 1 (0x22)
// Core Port Running 0 (0x4B) 
aiex.ipu.write32 { column = 0 : i32, row = 4 : i32, address = 0x340E0 : ui32, value = 0x4B222125 : ui32 }

// Events 4-7 monitored
// ------------------------
// Core Port Running 1 (0x4F)
// Lock stalls (0x1A)
// Lock acquire requests (0x2C)
// Lock release requests (0x2D)
aiex.ipu.write32 { column = 0 : i32, row = 4 : i32, address = 0x340E4 : ui32, value = 0x2D2C1A4F : ui32 }
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

<u>Example</u>
```
// Stream_Switch_Event_Port_Selection_0
// This is necessary to capture the Port_Running_0 and Port_Running_1 events
// Port 0 - Master/ID=1, Port 1 - Slave/ID=1
aiex.ipu.write32 { column = 0 : i32, row = 4 : i32, address = 0x3FF00 : ui32, value = 0x121 : ui32 }
```


### <u>Configure shimDMA</u>

The shimDMA needs to be configured to write the trace stream data to a valid location in DDR memory to be read by the host code. In the case of Ryzen AI, we can use a template like the following where the main parameters that need to be defined are the `buffer_length` and `buffer_offset`. 

* The `buffer_length` is the expected trace buffer size in bytes and should match what's expected from the host code when reading the trace buffer.
* The `buffer_offset` specifies in bytes where the trace buffer starts in the output buffer and occurs after the main output buffer ends. If the output buffer size in words is 65,536, then the buffer offset would be 4*65,536 = 262,144 bytes.
* `packet_id` - TODO for packet routing?
* `packet_type` - TODO for packet routing?`
* `column` - TODO shim tile column value?
* `column_num` - TODO number of columns in design?

<u>Example</u>
```
aiex.ipu.writebd_shimtile { bd_id = 3 : i32,
                            buffer_length = 16384 : i32,
                            buffer_offset = 262144 : i32,
                            enable_packet = 0 : i32,
                            out_of_order_id = 0 : i32,
                            packet_id = 0 : i32,
                            packet_type = 0 : i32,
                            column = 0 : i32,
                            column_num = 1 : i32,
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
                            ddr_id = 2 : i32,
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
                            use_next_bd = 0 : i32,
                            valid_bd = 1 : i32}
// Set start BD to our shim bd_Id (3)
aiex.ipu.write32 { column = 0 : i32, row = 0 : i32, address = 0x1D20C : ui32, value = 0x3 : ui32 }
```
