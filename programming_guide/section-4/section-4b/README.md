<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 4b - Trace</ins>

* [Section 4 - Vector programming & Peformance Measurement](../../section-4)
    * [Section 4a - Timers](./section-4a)
    * [Section 4b - Trace](./section-4b)
    * [Section 4c - Kernel vectorization](./section-4c)
    * [Section 4d - Automated vectorization](./section-4d)

In the previous [section-4a](../section-4a), we looked at how timers can be used to get an overview of application performance. However, for kernel programmers that want to optimize the AIE hardware to its fullest potential, being able to see how efficiently the VLIW core processors and data movers are running is important. As such, the AIEs are equipped with tracing hardware that provides a cycle accurate view of hardware events. More detailed specification of the AIE trace hardware can be found at [insert link here](insert_link).

Enabling trace support can be done in a few simple steps as outlined below:

## <u>Steps to Enable Trace Support</u>
1. Enable and configure trace hardware for a given AIE Tile
1. Configure testbench to read trace data and write it to a text file
1. Parse text file to generate a tracing json file
1. Open json file in a visualization tool like Perfetto
1. Debug Hints


## <u>1. Enable and configure trace hardware for a given AIE Tile</u>

Enabling tracing means configuring the trace hardware in the AIE array to monitor particular events and then routing those events through the stream switches to the shim DMA where we can write them to a buffer in DDR for post-run processing.

### <u>1a) Define trace event routes from tile to shimDMA</u>
#### <u>Circuit switched flows</u>

Trace event packets can be circuit switch routed or packet switch routed from the tiles to the shimDMA. An example of a simple circuit switch routing for the trace events from a compute tile to the shimDMA would be:

```
    flow(ComputeTile, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)
```

It is important to consider how many streams this routing will take and whether designs may experience stream routing congestion or not. While capturing trace events are non-intrusive (does not affect the performance of the AIE cores), the routing of these trace packets are not and need to be balanced in your design to prevent congestion.

#### <u>Packet switched flows</u>
To support packet switched flows, we need to declare packet flows and attach both a packet ID and packet type to the packets. This is needed to distinguish packets coming from different tiles types.(tile core, tile memory, memtiles, shimtiles). The designation of IDs and types are as follows:

| AIE trace block | packet type |
|-----------------|-------------|
| Tile core       | 0           |
| Tile memory     | 1           |
| Interface/ ShimTile       | 2           |
| MemTile         | 3           |

The packet IDs can be anything you want as long as they are globally unique to distinguish routes from one trace control from another. An example is shown below for two tiles with core and memory trace units routed. Note the flow ID used after the `packetflow` keyword. Also note that we want to set `keep_pkt_header = true` as we would like to keep the packet headers when they are moved to DDR so we can distinguish the packets during post-run parsing.

In IRON python bindings, we declare packet flows with the following syntax:
packetflow(flowID, Source Tile, Source Port Name, Source Port Channel, Destination Tile, Destination Port Name, Destination Port Channel, Keep Packet Header boolean)
```
packetflow(1, ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1, keep_pkt_hdr=True)
```
The first argument is the packet ID/ flow ID that uniquely identifies each packet flow. 
Then we have 3 arguments for the source and 3 for the destination. 
Tile Name - Previously defined tile name
Tile Port - Wire bundles for the port including WireBundle.Trace, WireBundle.DMA, WireBundle.North, etc. 
Tile Channel # - For a given port name, we sometimes have more than one channel such as DMA channel 0, and DMA channel 1. Trace ports only have port 0 for AIE2.

MLIR examples are similar and are includeed below:
```
packetflow(1) { 
    aie.packet_source<%tile02, Trace : 0> // core trace
    aie.packet_dest<%tile00, DMA : 1>
} {keep_pkt_header = "true"}
```
The second necessary component to support trace configuraton is programming the trace control register for each tile that we want to enable tracing for. We have abstracted this with the python wrapper fucntion `configure_simple_tracing_aie2` which is described in more detail in [python/utils](../../../python/utils).

### <u>1b) Configure Trace Hardware for a Given Tile</u>
The configuraton of trace hardware for a given tile core, tile memeory, memtile and shimtile leverages the python utility `utils/test.py`. More detail on how to use this python utility and what configuration are used can be found in [python/utils](../../../python/utils/).

## <u>2. Configure testbench to read trace data and write it to a text file</u>

Once the trace hardware is configured, we want the testbench/ host code to read the trace data from DDR and write it out to a text file for post-run processing. In the case of a python code in a jupyter notebook, we can do this with a few python calls.

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

## <u>3. Parse text file to generate a tracing json file</u>
Once the trace text file is generated (trace.txt), we use python trace parsers to interpret the trace values and generate a json trace file for visualization. This leverages the parase python scripts under [programming_examples/utils](../../../programming_examples/utils/). Please click the link for more information about the parse python scripts and how they are used. 

## <u>4. Open json file in a visualization tool like Perfetto</u>
Open the https://ui.perfetto.dev in your browser and then open up the generated json trace file generated in step #3. You can zoom in the waveform visualizer with a,s,w,d keys. 

## <u>Debug Hints</u>
* If you are getting 0's in your trace outputs. Check these things:
    * Buffer offset for the DMA is the right size (based on output buffer size)
    * The correct tile is being routed to the the correct shim DMA. It's easy in a multi core design to route the wrong tile, espeicall if the tile symbols/ names are very similar or confusing.
    * Check matching packet IDs for packet-routed flows. The packet flow ID must match the configured ID value in Trace Control 1 register or else the packets don't get routed.