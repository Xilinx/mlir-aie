<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 7 - Communication via flows (broadcast)</ins>

Working with circuit switch and packet switch broadcast at the flow level involves different sets MLIR-AIE operators and a deeper understanding of how the underlying hardware behaves. Lowering down even further to the [switchbox](../switchbox) level provides even more understanding and finer granularity of control over the AI Engine array. Let's first take a look at how we work with broadcast at the flow level.

## <ins>Circuit switch broadcast</ins>
For circuit switch connections, broadcast is supported by simply making logical connections between the same ports. For example, if we have a single source routing to 3 destinations, it may look like:
```
AIE.flow(%tile14, DMA: 0, %tile34, DMA:0)
AIE.flow(%tile14, DMA: 0, %tile35, DMA:1)
AIE.flow(%tile14, DMA: 0, %tile36, DMA:0)
```
Here, the same tile DMA and channel combination is used as the source. The router will then recognize a broadcast scenario and make the appropriate shared connections.
> Note that the way circuit switch broadcast works is where the source pushies data simultaneously to all destinations. If any destination cannot receive the data and exerts backpressure, the source stops sending, even if another destination can still receive data. Knowing that this occurs is important when designing communication scenarios.

## <ins>Packet switch broadcast</ins>

The packet switch case is much more robust as we can do both one-to-many connections as well as many-to-one. This is because packets have natural boundaries where the data begins and ends and the stream switch is configured to do a round-robin arbitration between ports sharing a single stream. Because packets are labeled with packet type and packet IDs, the broadcast definition will likewise use these designations to help guide the stream switch configurations. A simple example configuration where we route all packets from a source with a certain packet ID to a set of destination (one-to-many) is shown below: 
```
 AIEX.broadcast_packet(%tile14, DMA: 0) {
      AIEX.bp_id(0xD) {
        AIEX.bp_dest<%tile34, DMA: 1>
        AIEX.bp_dest<%tile35, DMA: 1>
      }
      ...
    }
```
Here, we use `AIEX.broadcast_packet` operator to denote the definition of packet flows from a given tile and DMA + channel (in this case, tile(1,4) and tileDMA MM2S channel 0). We then define the flows for every defined packet ID since packet IDs refer to a particular flow. For example, packet ID 3 might go one particular route, while packet ID 4 might go another route (or the same route). In this example, we define the flow for packet ID 0xD and indicate that it goes to both tile(3,4) and tile (3,5) to the associated tile DMA (S2MM, both channel 1). This list of destinations can be just one or as many as desired. The ... in this code snippet indicates that we can continue to define more flows for different packet IDs.

> **NOTE**: We have used the AIEX dialect which is an experimental extension of the MLIR-AIE dialect where new dialect operations are being developed. More details of existing `AIEX` dialect operations can be found [here](https://xilinx.github.io/mlir-aie/AIEXDialect.html).

If broadcast is not needed, you could use the simpler `AIE.packet_flow`. This functions very similarly to `AIE.flow` with two differences: (1) packet ID is associated with a packet flow, (2) We wrap the source and destination ports with `AIE.packet_source` and `AIE.packet_dest` like:
```
AIE.packet_flow(0xD) {
  AIE.packet_source<%tile14, "Core" : 0>
  AIE.packet_dest<%tile34, "Core" : 0>
}
```
However, because this can be done with the `AIE.broadcast_packet` operation with a single destination, we can just use `AIE.broadcast_packet` for the single source to single destiantion case as well.

The configurability of the stream switch is more nuanced that is indicated with these MLIR operations and that nuance is exposed with lower level operators which can be further explored in [tutorial-7 - switchbox](../switchbox).

## <ins>Tutorial 7 Lab </ins>

1. Read through the [/flow/aie.mlir](aie.mlir) design. What is the packet ID used in this example? <img src="../../images/answer1.jpg" title="0xD" height=25>

2. Add a third block at tile(3,6) and broadcast to this new block as well.

3. Change the tileDMA behavior by adding a new bd which pushes data with packet ID 0xE. If we want to broadcast this new packet ID to tile(3,6) but leave the original packet ID routed to tile(3,4) and tile(3,5), how would we do that?
