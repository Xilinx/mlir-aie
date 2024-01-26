# Tracing

## <u>Outline</u>
- [1. Manually setting up trace configuration in a given design](#1.-Manually-setting-up-trace-configuration-in-a-given-design)
- [2. Configuring host code/ XRT to write trace values in hex format into a text file](#2.-Configuring-host-code/-XRT-to-write-trace-values-in-hex-format-into-a-text-file)
- [3. Parse text file to generate a tracing json file](#3.-Parse-text-file-to-generate-a-tracing-json-file)
- [4. Open json file in a visualization tool like https://ui.perfetto.dev/](#4.-Open-json-file-in-a-visualization-tool-like-https://ui.perfetto.dev/)

The current methodology for enabling tracing invovles 4 main steps.
1. Manually setting up trace configuration in a given design
2. Configuring host code/ XRT to write trace values in hex format into a text file
3. Parse text file to generate a tracing json file
4. Open json file in a visualization tool like https://ui.perfetto.dev/

## 1. Manually setting up trace configuration in a given design

Enabling tracing means configuring the trace hardware in the AIE array to monitor particular events and then routing those events through the stream switches to the shim DMA where we can write them to a buffer in DDR for post-run processing. This configuration occurs within the .mlir source file (or .py source if using python bindings TODO).

### <u>Define trace event flows from tile to shimDMA</u>
Trace event packets can be circuit switch routed or packet switch routed from the tiles to the shimDMA. An example of a simple circuit switch routing for the trace events from a single tile in column 0, row 5 (`tile05`) to the shimDMA in column 0 would be:

`aie.flow(%tile05, "Trace" : 0, %tile00, "DMA" : 1)`

It is important to consider how many streams this routing will take and whether designs may experience stream routing congestion or not. While capturing trace events are non-intrusive (does not affect the performance of the AIE cores), the routing of these events are not and need to be balanced to prevent congestion.

***Insert packet switch routing example***

Within the `func.func @sequence` block, we add a set of `aiex.ipu.write32` to configure the tile trace settings, the shimDMA, 

### <u>Configure tile trace settings</u>
For a give AIE2 tile, we configure the trace control for the tile core and tile memory separately. There are 4 registers we generally use to configure the trace behavior. 2 are for configuring the trace general control and the other 2 are to specify which events our tile's trace hardware is monitoring.


| Config Register | Address | Field | Bits | Reset Value | Description |
|-----------------|---------|-------|------|-------------|-------------|
| Trace Control 0 | 0x340D0 | Stop Event | [30:24], 0xNN------ | 0 | Event to stop trace capture | 
| Trace Control 0 | 0x340D0 | Start Event | [22:16], 0x--NN---- | 0 | Event to start trace capture |
| Trace Control 0 | 0x340D0 | Mode | [1:0], 0x-------N | 0 | Trace mode. 00=event-time, 01=event-PC, 10=execution |
| Trace Control 1 | 0x340D4 | Dest Packet Type | [14:12], 0x----N--- | 0 | Detination trace packet - packet type |
| Trace Control 1 | 0x340D4 | Dest Packet ID | [4:0], 0x------NN | 0 | Detination trace packet - packet ID |

<u>Example</u>
```
// Start event = 1, Stop event = 0, Mode = event-time
aiex.ipu.write32 { column = 0 : i32, row = 4 : i32, address = 0x340D0 : ui32, value = 0x10000 : ui32 }
```

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

There's is an extensive lists of trace events but we will describe a few key ones here.
| Some common events | event ID | dec value |
|--------------------|----------|-----------|
| Ture                       |0x01| 1 |
| Stream stalls              |0x18| 24 |
| Core Instruction Events 1  |0x22| 34 |
| Core Instruction Events 0  |0x21| 33|
| Vector Instructions (e.g. VMAC, VADD, VCMP) |0x25|  37 |
| Lock acquire requests      |0x2C|  44 |
| Lock release requests      |0x2D|  45 | 
| Lock stall                 |0x1A|  26 |
| Core Port Running 1        |0x4F|  79 |
| Core Port Running 0        |0x4B|  75 | 


<u>Example</u>
```
// Events 0-3 monitored
// ------------------------
// Vector instrucitons (0x25)
// Core Instruction Events 0 (0x21)
// Core Instruction Events 1 (0x22)
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


Some configurations like the Port Running 0/1 events are further configured by a secondary configuration register. In this case, we route the port activity from the stream switch to 0 or 1.
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

The shimDMA needs to be configured to write the trace stream data to a valid location in DDR memory to be picked up by the host code. In the case of Ryzen AI, we can use a template like the following where the main parameters that get updated is the `buffer_length` and `buffer_offset`. 

* The `buffer_length` is the expected trace buffer size in bytes.
* The `buffer_offset` occurs after the output buffer in bytes. If the output buffer size in words is 65,536, then the buffer offset would be 4*65,536 = 262,144 bytes.
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
```

## 2. Configuring host code/ XRT to write trace values in hex format into a text file

Once the trace hardware is configured, we want the host code to read the trace data in DDR and write it out to a text file for post-run processing. In the case of a python code in a jupyter notebook, we can do this with a few python calls.


### Extracting trace from output buffer and writing values out to a text file
This section of code that runs application and reads the output buffer into `output`. We call helper functions to extract the trace data and write it out to a file. One parameter is the trace buffer size (`trace_size`) which we declare here and should match the [shimDMA `buffer_length` above](#configure-shimdma).
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
Since `out_buf_shape` is used during extraction, we modify the output buffer size by the `trace_size`
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


### Extract Trace Helper Function
Extract trace data from output buffer data where trace data is appended after the output buffer. 
* `trace_size` - Must specify this in bytes so we extract the right amount of data
```
def extract_trace(out_buf, out_buf_shape, out_buf_dtype):
    trace_size_words = trace_size//4
    out_buf_flat = out_buf.reshape((-1,)).view(np.uint32)
    output_prefix = out_buf_flat[:-trace_size_words].view(out_buf_dtype).reshape(out_buf_shape)
    trace_suffix = out_buf_flat[-trace_size_words:]
    return output_prefix, trace_suffix
```
### Write Trace Out Helper Function
Write each trace packet as a hex string, one per line.
```
def write_out_trace(trace, file_name):
    out_str = "\n".join(f"{i:0{8}x}" 
                        for i in trace
                        if i != 0)
    with open(file_name, 'w') as f:
        f.write(out_str)
```


## 3. Parse text file to generate a tracing json file

Once the trace data is stored in a text file, we want to parse it to generate waveform. There are two paths for this at the moment, one is a custom parser `parse_trace.py` that will generate a .json file which we can open in Perfetto to view and navigate the waveforms. The other is to use the emitIR parser `parse_eventiR.py` which will also generate a .json file. In order to use this parser, we must first convert our trace data into eventIR format using the Vitis hwfrontend parser which is used by aiesimulator. Both flows are described below:

### a) Custom trace data parser --> .json
To call our custom parser, we need the following files:
* trace data text file - Generated during the running of our python host code/ jupyter notebook
* source mlir - This is needed to parse what events and tiles were are monitoring to generate labels for our waveform visualizer
* column shift - This specifies how much the actual design was shifted from the default position when it was scheduled and called. The reason we need this is becuase even if our design is targeting column 0, the actual loading and execution of the design may place it in column 1, 2, 3 etc. We account for this shift since the parser needs to match the column location of the mlir source with the column of the generated trace data. Usually 2 is the right value. NOTE - the underlying tools currently default to at column 1 to avoid using column 0 on Ryzen AI since that column does not have a shimDMA and is therefore avoided at the moment.

From the notebook folder, where the resnet designs are run, we should make sure we have a trace output folder created (by default, we use `traces`). Then we can run the following command.
```
../../../utils/parse_trace.py --filename traces/bottleneck_cifar_split_vector.txt --mlir ../bottleneck_block/bottleneck_cifar_split_vector/aieWithTrace.mlir --colshift 2 |& tee mytrace.json
```

### b) Vitis hwfrontend + parser --> .json

1. Create a dummy file (`.target`) in the current directory with the content 'hw'
2. Create a template json with the matching tile position and events - <custom config>.json
3. Prepend 0x in front of all event trace packet in trace text file - <0x trace text file>
4. Modify trace text file of possible bugs (see below)
5. Run Vitis frontend parser to generate event IR.
6. Run custom eventIR-to-json parser script (`parse_eventIR.py`)

After step #1, we need a template json file that matches the position and event list of our trace. This file should ideally be auto-generated but an example version of this file can be found in reference_designs/ipu-xrt/resnet/notebook/traces/
```
hwfrontend --trace <0x trace text file> --trace_config <custom config>.json --pkg-dir . --outfile <output text file>
```
#### <u>NOTE</u>: Some errors that have cropped up. 
<u>Bug 1</u>: For case where start event is 1. The trace output seems to have a few packets with just `0xdbffdbff` data. These seem to give the following error and needs to have those packets deleted up to the start packet.
```
CRITICAL WARNING: [hwanalyze 77-5570] trace_out_hex3:1 Start Frame for Tile(2, 4) Module: cm looks to be missing as trace configuration is not available.
```
<u>Bug 2</u>: If the start timer value is too large, it reports an error:
```
WARNING: [hwanalyze 77-5569] trace_out_hex2:1 Overrun Tile(2, 4) Module: cm. Hence further decoding of this particular module will be skipped.
```
So reducing the start frame from something like:
```
0xf4000000
0x00a93c62
to 
0xf0000000
0x0005d0f7
```
which reduces the timer from 11,091,042 cycles to 381,175 seems to fix it.

5. Run eventIR parse script to generate json file for waveforms
```
../../../utils/parse_eventIR.py --filename <output text file> --mlir ../bottleneck_block/bottleneck_cifar_split_vector/aieWithTrace.mlir --colshift 2 |& tee mytrace.json
```


## 4. Open json file in a visualization tool like https://ui.perfetto.dev/
Open the https://ui.perfetto.dev in your browser and then open up the trace that was generated in the previous step. 
