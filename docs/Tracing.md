# Tracing

## <u>Outline</u>
- [1. Manually setting up trace configuration in a given design](#1.-Manually-setting-up-trace-configuration-in-a-given-design)
- [2. Configuring host code/ XRT to write trace values in hex format into a text file](#2.-Configuring-host-code/-XRT-to-write-trace-values-in-hex-format-into-a-text-file)
- [3. Parse text file to generate a tracing json file](#3.-Parse-text-file-to-generate-a-tracing-json-file)
- [4. Open json file in a visualization tool like Perfetto](#4.-Open-json-file-in-a-visualization-tool-like-Perfetto)
- [Debug Hints](Debug-Hints)

## 1. Manually setting up trace configuration in a given design

Enabling tracing means configuring the trace hardware in the AIE array to monitor particular events and then routing those events through the stream switches to the shim DMA where we can write them to a buffer in DDR for post-run processing. This configuration occurs within the .mlir source file (or .py source if using python bindings TODO).

### <u>Define trace event flows from tile to shimDMA</u>
#### <u> Circuit switched flows </u>

Trace event packets can be circuit switch routed or packet switch routed from the tiles to the shimDMA. An example of a simple circuit switch routing for the trace events from a single tile in column 0, row 5 (`tile05`) to the shimDMA in column 0 would be:

`aie.flow(%tile05, "Trace" : 0, %tile00, "DMA" : 1)`

It is important to consider how many streams this routing will take and whether designs may experience stream routing congestion or not. While capturing trace events are non-intrusive (does not affect the performance of the AIE cores), the routing of these events are not and need to be balanced to prevent congestion.

#### <u> Packet switched flows </u>
To support packet switched flows, we need to delcare packet flows and attach both an packet ID and packet type to the packets. This is needed to distinguish packets from different tiles cores and tile memories. The designation of IDs and types are somewhat aribtrary but we can use the following for types:

| AIE trace block | packet type |
|-----------------|-------------|
| Tile core       | 0           |
| Tile memory     | 1           |
| Interface       | 2           |
| MemTile         | 3           |

The packet IDs can be anything you want as long as they are globally unique to distinguish routes from one trace control from another. An example is shown below for two tiles with core and memory trace units routed. Note the flow ID used after the `packet_flow` keyword. Also note that we want to set `keep_pkt_header = true` as we would like to keep the packet headers when they are moved to DDR so we can distinguish the packets during post-run parsing.
```
aie.packet_flow(1) { 
    aie.packet_source<%tile02, Trace : 0> // core trace
    aie.packet_dest<%tile00, DMA : 1>
} {keep_pkt_header = "true"}

aie.packet_flow(2) { 
    aie.packet_source<%tile02, Trace : 1> // memory trace
    aie.packet_dest<%tile00, DMA : 1>
} {keep_pkt_header = "true"}

aie.packet_flow(3) { 
    aie.packet_source<%tile03, Trace : 0> // core trace
    aie.packet_dest<%tile00, DMA : 1>
} {keep_pkt_header = "true"}

aie.packet_flow(4) { 
    aie.packet_source<%tile03, Trace : 1> // memory trace
    aie.packet_dest<%tile00, DMA : 1>
} {keep_pkt_header = "true"}
```
The second necessary component to support packet routing is in the trace contorl configuration which we will highlight in the next section.

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

## 2. Configuring host code/ XRT to write trace values in hex format into a text file

Once the trace hardware is configured, we want the host code to read the trace data from DDR and write it out to a text file for post-run processing. In the case of a python code in a jupyter notebook, we can do this with a few python calls.


### Extracting trace from output buffer and writing values out to a text file
This section of code runs the application and reads the output buffer into the variable `output`. We call some helper functions to extract the trace data and write it out to a file. One parameter that is important to define is the trace buffer size (`trace_size`) which we declare here and should match the shimDMA `buffer_length` [above](#configure-shimdma).
```
trace_size = 16384    # in bytes

...

app.run()
output = app.buffers[4].read()
if enable_trace:
    output, trace = extract_trace(output, shape_out, dtype_out)
    write_out_trace(trace, trace_file)
```
### Modification for `setup_aie` to update output buffer shape (`out_buf_shape`)
Since `out_buf_shape` is used during `extract_trace`, we modify the output buffer size by the `trace_size`
```
def setup_aie(xclbin_path, insts_path, 
              in_0_shape, in_0_dtype,
              in_1_shape, in_1_dtype, 
              out_buf_shape, out_buf_dtype,
              enable_trace=False,
              kernel_name="MLIR_AIE"):
    app = xrtutils.AIE_Application(xclbin_path, insts_path, kernel_name)
    app.register_buffer(2, shape=in_0_shape, dtype=in_0_dtype)
    app.register_buffer(3, shape=in_1_shape, dtype=in_1_dtype)
    if enable_trace:
        out_buf_len_bytes = np.prod(out_buf_shape) * np.dtype(out_buf_dtype).itemsize
        out_buf_shape = (out_buf_len_bytes + trace_size, )
        out_buf_dtype = np.uint8
    app.register_buffer(4, shape=out_buf_shape, dtype=out_buf_dtype)
    return app
```


### "Extract Trace" helper function
This helper function extracts trace data from the output buffer where the trace data is appended after the output buffer. 
* `trace_size` - Defined earlier (in bytes) and needs to be specified outside the function so we extract the right amount of data
```
def extract_trace(out_buf, out_buf_shape, out_buf_dtype):
    trace_size_words = trace_size//4
    out_buf_flat = out_buf.reshape((-1,)).view(np.uint32)
    output_prefix = out_buf_flat[:-trace_size_words].view(out_buf_dtype).reshape(out_buf_shape)
    trace_suffix = out_buf_flat[-trace_size_words:]
    return output_prefix, trace_suffix
```
### "Write Trace Out" helper function
This helper function writes each trace packet as a hex string, one per line, out to a text file.
```
def write_out_trace(trace, file_name):
    out_str = "\n".join(f"{i:0{8}x}" 
                        for i in trace
                        if i != 0)
    with open(file_name, 'w') as f:
        f.write(out_str)
```


## 3. Parse text file to generate a tracing json file

Once the trace data is stored in a text file, we want to parse it to generate waveform json file. There are 2 flows to do this at the moment, one is a custom parser `parse_trace.py` that will generate a .json file which we can open in Perfetto to view the waveforms. The other is to use the eventIR parser `parse_eventiR.py` which will also generate a .json file. In order to use this second parser, we must first convert our trace data into eventIR format using the Vitis hwfrontend parser which is used by aiesimulator. The goal of this second flow is to leverage the existing trace packet parsing from aiesimulator. Both flows are described below:

### a) Custom trace data parser --> .json
To call our custom parser, we need the following files:
* `trace data text file` - This is generated during the running of our python host code/ jupyter notebook
* `source mlir` - This is needed to parse what events and tiles we are monitoring to generate labels for our waveform visualizer
* `column shift` - This specifies how much the actual design was shifted from the default position when it was scheduled and called. The reason we need this is becuase even if our design is configured for column 0, the actual loading and execution of the design may place it in column 1, 2, 3 etc. We account for this shift since the parser needs to match the actual column location of the generated trace data. Usually 2 is the right value. NOTE - the underlying tools currently default to column 1 to avoid using column 0 on Ryzen AI since that column does not have a shimDMA and is therefore avoided at the moment.

From the notebook folder, where the resnet designs are run, we should make sure we have a trace output folder created (by default, we use `traces`). Then we can run the following command.
```
../../../utils/parse_trace.py --filename traces/bottleneck_cifar_split_vector.txt --mlir ../bottleneck_block/bottleneck_cifar_split_vector/aieWithTrace.mlir --colshift 2 > trace.json
```

### b) Vitis hwfrontend + parser --> .json

NOTE: This flow is still being developed and many of the required steps at this moment should likely be rolled into the script directly. For now, it's probably better to just use the custom parser flow and only use this flow to compare results.

1. Create a dummy file (`.target`) in the current directory with the file content 'hw'
2. Create a template json with the matching tile position and events - <custom config>.json
3. Prepend 0x in front of all event trace packet in trace text file - <0x trace text file>
4. Modify trace text file to workaround possible bugs (see below)
5. Run Vitis frontend parser to generate an event IR text file.
6. Run custom eventIR-to-json parser script (`parse_eventIR.py`)

Step #1 is needed by the `hwfrontend` tool and is just a hidden file that has `hw` in the first line. 

For step #2, we need a template json file that matches the position and event list of our trace. This file should ideally be auto-generated but an example version of this file can be found in `reference_designs/ipu-xrt/resnet/bottleneck_block/bottleneck_cifar_split_vector/traces/bottleneck_cifar_split_vector.json`

In step #3, we need to prepend the trace text file data with `0x` because that's what `hwfrontend` expects. Then in step #4, there are currently a few bugs that we may need to work around by editing the trace text file further as described below.

<u>Workaround 1</u>: For case where the start event is 1 or maybe in general, the trace output might have a few packets with just `0xdbffdbff` data. These seem to give the following error and needs to have those packets deleted up to an actual valid event packet.
```
CRITICAL WARNING: [hwanalyze 77-5570] trace_out_hex3:1 Start Frame for Tile(2, 4) Module: cm looks to be missing as trace configuration is not available.
```
<u>Workaround 2</u>: If the start timer value is too large, it reports an error:
```
WARNING: [hwanalyze 77-5569] trace_out_hex2:1 Overrun Tile(2, 4) Module: cm. Hence further decoding of this particular module will be skipped.
```
So reducing the start frame from something like:
```
0xf4000000
0x00a93c62
```
to
```
0xf0000000
0x0005d0f7
```
which reduces the timer from 11,091,042 cycles to 381,175 seems to fix it.
Step #5 is running the aiesimulator frontend parser which generates the eventIR text file.
```
hwfrontend --trace <0x trace text file> --trace_config <custom config>.json --pkg-dir . --outfile <output text file>
```

Step #6 is running the custom eventIR-to-json parser script (`parse_eventIR.py`) to generate the json file.
```
../../../utils/parse_eventIR.py --filename <output text file> --mlir ../bottleneck_block/bottleneck_cifar_split_vector/aieWithTrace.mlir --colshift 2 > trace_eventIR.json
```
Note that there is a sample of these steps for both the custom parser and the eventIR parser in the Makefile for bottleneck_cifar_split_vector located at `reference_designs/ipu-xrt/resnet/bottleneck_block/bottleneck_cifar_split_vector/traces/Makefile`. It doesn't have commands for step 3 and 4 which would need to be done by hand.

## 4. Open json file in a visualization tool like Perfetto
Open the https://ui.perfetto.dev in your browser and then open up the trace that was generated in the previous steps. You can zoom in the waveform visualizer with a,s,w,d keys. 

## <u>Debug Hints</u>
* If you are getting 0's in your trace outputs. Check these things:
    * Buffer offset for the DMA is the right size (based on output buffer size)
    * The correct tile is being routed to the the correct shim DMA. It's easy in a multi core design to route the wrong tile, espeicall if the tile symbols/ names are very similar or confusing.
    * Check matching packet IDs for packet-routed flows. The packet flow ID must match the configured ID value in Trace Control 1 register or else the packets don't get routed.