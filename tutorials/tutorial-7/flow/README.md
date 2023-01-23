<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 7 - communication (broadcast)</ins>

Now that we've seen how point to point communciation is handled, we need to introduce the concept of broadcast. This is necessary to allow data from a single source to be sent to multiple destinations. This is also important because with a fixed set of routing resources (e.g. channels), we need to be efficient in how we move data to avoid congestion. Broadcast then is our one-to-many communication tool. Many-to-one is also possible but only in the packet routing case. For circuit switch routing, we would need some kind of aggregator that decides when connections can be switched, which packet routing already supports.

## <ins>Circuit switch broadcast</ins>
For circuit swtich connections, broadcast is supported by simply making logical connections between the same ports. For exmaple, if we have a single source routing to 3 destinations, it may look like:
```
AIE.flow(%tile14, DMA: 0, %tile34, DMA:0)
AIE.flow(%tile14, DMA: 0, %tile35, DMA:1)
AIE.flow(%tile14, DMA: 0, %tile36, DMA:0)
```
Here, the same tile DMA and channel combination is used as the source. The router will then recognize a broadcast scenario and make the appropriate shared connections.
> Note that circuit switch broadcast functions where the source is pushing data simulatenously to all destinations. If any destination cannot receive the data and exerts backpressure, the source stops sending, even if another destination can still receive data. This type of backpressure is automatic but is important when designing communication scenarios.

## <ins>Packet switch broadcast</ins>

The packet swtich case is much more robust as we can not do both one-to-many connections as well as many-to-one. This is because packets have natural boundaries where the data begins and ends and the stream swtich is configured to do a round-robin arbitration between ports sharing a single stream. Because packets are labeled with packet type and packet IDs, the broadcast definition will likewise use these desginations to help guide the stream switch configurations. A simple example configuration where we route all packets from a source with a certain packet ID to a set of destination (one-to-many) is shown below: 
```
 AIE.broadcast_packet(%tile14, DMA: 0) {
      AIE.bp_id(0xD) {
        AIE.bp_dest<%tile34, DMA: 1>
        AIE.bp_dest<%tile35, DMA: 1>
      }
      ...
    }
```
Here, we use `AIE.broadcast_packet` opeator to denote the definition packet flows from a given tile and DMA + channel (in this case, tile(1,4) and tileDMA MM2S channel 0). We then define the flows for every defined packet ID since packet IDs refer to a particular flow. For example, packet ID 3 might go one particular route, while packet ID 4 might go another route (or the same route). In this example, we define the flow for packet ID 0xD and indicate that it goes to both tile(3,4) and tile (3,5) to the associated tile DMA (S2MM, both channel 1). This list of destination can be just one or as many as desired. The ... in this code snippet indicates that we can continue to define more flows for different packet IDs.

The configurabiliy of the stream switch is more nuanced that is indicated with these MLIR operations and that nuance is exposed with lower level operators which can be further explored in [tuorial 7a](./tutorial-7a).

## <ins>Tutorial 7 Lab </ins>

1. Read through the [/flow/aie.mlir](aie.mlir) design. What is the packet ID used in this example? <img src="../../images/answer1.jpg" title="0xD" height=25>

2. Add a third block at tile(3,6) and broadcast to this new block as well.

3. Change the tileDMA behavior by adding a new bd which pushes data with packet ID 0xE. If we want to broadcast this new packet ID to tile(3,6) but leave the original packet ID routed to tile(3,4) and tile(3,5), how would we do that?
