<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Trace Details (Unplaced and Placed designs)</ins>

* [Section 4 - Performance Measurement & Vector Programming](../../section-4)
    * [Section 4a - Timers](../section-4a)
    * [Section 4b - Trace](../section-4b)
    * [Section 4c - Kernel Vectorization and Optimization](../section-4c)

-----

In [section-4b](../section-4b), we introduced how trace is enabled in our high-level IRON python designs. These designs are unplaced and relies on the tools to decide on a placement for the tiles. Because of this, certain assumptions and limitation are introduced to help simplify the trace configuration. These include the following:

## Trace Considerations for High-level IRON Python
* Only core tiles are currently able to be traced in high-level IRON python since we trace workers and workers are attached to core tiles. If you want to trace mem tiles and shim tiles, see the next section on how to do that in close-to-metal IRON python (adding this to high level IRON python TBD)
* A shim tile is needed to route trace packets. We assume all designs have an objfifo where the consumer endpoint is on a shimtile. High-level IRON currently employs a convenience algorithm to search for the first objfifo going out of the AIE array and configures it for trace on channel 1. If that shim dma already has channel 1 configured for something else, a conflict can occur
* We assign packet flows with incrementing route IDs for each flow between a tile and the shim dma. There is not currently a step to resolve these route IDs with ones used by other kinds of packet flows. At the moment, objfifos only operate in circuit switched flows so this is an not an immediate issue. 

## <u>1. Enable and configure AIE trace units for close-to-metal IRON Python ([aie2_placed.py](./aie2_placed.py))</u>

Enabling tracing means (1a) configuring the trace units for a given tile and then (1b) routing the generated events packets through the stream switches to the shim DMA where we can write them to a buffer in DDR for post-runtime processing. In close-to-metal IRON python, these steps require a more explicit declaration and descrbied below:

### <u>(1a) Configure trace units for an AIE tile</u>
The first necessary component for trace configuration is setting the right values for the trace control registers for each tile that we want to enable tracing for. In addition, the generated trace packets will need to be routed to a shimDMA and then written to an inout buffers so the packet data can be written to DDR. We have abstracted these two steps with the python wrapper function `configure_packet_tracing_aie2` which is in [python/utils/trace.py](../../../python/utils/trace.py) and is described in more detail in the [README](../../../python/utils) under `python/utils`. An example of how this function is used is shown below for quick reference:
```python
    trace_utils.configure_packet_tracing_aie2(tiles_to_trace, ShimTile, opts.trace_size)
```
The arguments for this example are:
* *tiles_to_trace* - array of tiles we want to trace (including mem tiles and shim tiles)
* *ShimTile* - shim tile that the trace is routed to for writing to DDR
* *opts.trace_size* - the trace buffer size in bytes

This block is defined within the sequence definition for `@runtime_sequence` where we define the shimDMA data movement to the inout buffers.
> **Note** This convenience python wrapper abtracts a number of sub-steps for configuring the trace unit in each tile and the shimDMA for writing to DDR. This uses packet switched routing to move the trace packets as opposed to circuit switched routing. More details on these sub-steps can be found in the [README](../../../python/utils) under `python/utils`.

Configuring the trace units with `configure_packet_tracing_aie2` should be declared at the beginning of the `@runtime_sequence` so the trace mechanisms are in place prior to any data being transferred from/ to DDR. At the end of the `@runtime_sequence` we add the following convenience python function to end the trace collection.
```python
    trace_utils.gen_trace_done_aie2(ShimTile)
```
This is necessary so the trace units flush intermediate trace packets.

### <u>(1b) Define trace routes from tiles to shimDMA</u>
Once the trace units and shimDMA are configured, we need to define how the trace packets are routed from the tiles to the Shim tile. This is done by declaring packet switched flows using the convenince function `configure_packet_tracing_flow` defined in [python/utils/test.py](../../../python/utils/test.py) and described in more detail in the [README](../../../python/utils) under `python/utils`. This function is declared in the main body of our design as shown below:
```python
    trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)
```
The arguments for this example are:
* *tiles_to_trace* - array of compute tiles we want to trace
* *ShimTile* - shim tile that the trace is going out to

> **Note** The synchronization of this function with the previous `configure_packet_tracing_aie2` is important because we track the flow IDs and bd numbers of each configured trace. Do not mix and match these with circuit switched routing as they are intended to work together as pair. 


## <u>Exercises</u>
1. We can try building our close-to-metal IRON python design now. Run `make clean; make use_placed=1 trace`. This compiles the placed design, generates a trace data file, and runs `prase_trace.py` to generate the `trace_4b.json` waveform file.

    Note that many of the designs under [programming_examples](../../../programming_examples/) have both an high-level IRON python version and a close-to-metal IRON python version, otherwise known as the placed version. Invoking make with the `use_placed=1` is a common way to build these versions of the design.

## <u>2. Customizing Trace Behavior</u>
The wrapper python function `configure_packet_tracing_flow` abstracts much of the configuration of trace by making some assumptions of the desired trace behavior. Some of those assumptions are listed below along with how to customize them further.

1. Additional configuration arguments for `configure_packet_tracing_flow` as descrbied in [utils/python](../../../python/utils)

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

    An example use case to customize coretile events and map it to a 3rd XRT buffer instead of the 5th XRT buffer would be:
    ```python
    trace_utils.configure_packet_tracing_aie2(
        tile_to_trace
        ShimTile, 
        opts.trace_size,
        ddr_id = 5,
        coretile_events = [ 
                trace_utils.CoreEvent.INSTR_EVENT_0,
                trace_utils.CoreEvent.INSTR_EVENT_1,
                trace_utils.CoreEvent.INSTR_VECTOR,
                trace_utils.CoreEvent.MEMORY_STALL,
                trace_utils.CoreEvent.STREAM_STALL,
                trace_utils.CoreEvent.LOCK_STALL,
                trace_utils.CoreEvent.ACTIVE,
                trace_utils.CoreEvent.DISABLED]
    )
    ```

-----
[[Prev]](../section-4a) [[Up]](../../section-4b) [[Next]](../section-4c)
