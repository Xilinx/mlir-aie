<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 6 - communication (packet routing)</ins>

We already looked at how we can use stream switches and tile DMAs to communicate data between tiles that are non-adjacent/ far apart in [tutorial-4](../tutorial-4). There we discussed how routes in a switchbox can be configured in circuit switch mode or packet switch mode and showed examples of circuit switch mode routing. Packet switch mode is another communication format and will will examine it more closely in this tutorial.

## <ins>Packets</ins>
Packets are central to packet-switched routing and provide the ultimate flexiblity for runtime data steering. A DMA transfer is converted into data packet with a prepended packet header and a signal marker for the end of packet. It is then forwarded from switchbox to switchbox along routes much like a circuit switched flow. The difference is that each switchbox along the flow can be configured to know where packets with a predefined packet ID is supposed to go and can forward packets, drop packets or strip the header from packets when the packet reaches its final destination. 

## <ins>Packet Flow</ins>
To convert a circuit switched flow into a packet switched one, we use a `AIE.packet_flow` instead of the circuit switched flow operator (`AIE.flow`) and we configure the source tile DMA to generate packets with the `AIE.dmaBdPacket` operation.

The packet flow operator is declared much like a regular flow with the following syntax:
```
AIE.packet_flow($packet_id) {
    AIE.packet_source<$tile, $bundle : $channel>
    AIE.packet_dest<$tile, $bundle : $channel>
}
```
An example of this is:
```
AIE.packet_flow(0xD) {
    AIE.packet_source<%tiler14, DMA : 0>
    AIE.packet_dest<%tile34, DMA : 1>
}
```
`$packet_id`: 5-bit unsigned packet ID that this flow is configured as

Much like iin `AIE.flow`, the %tile, %bundle and %channel represent valid endpoints specific to the architecture. 

In the above flow syntax, valid bundle names and channels are listed below: 
| Bundle | Channels (In) | Channels (Out) |
|-------|---|---|
| DMA   | 2 | 2 |
| Core  | 2 | 2 |
| West  | 4 | 4 |
| East  | 4 | 4 |
| North | 4 | 6 |
| South | 6 | 4 |
| PLIO  | 2?| 2?|

> Note that be default, packet flows with a DMA destination will configure the destination tile swtichbox to strip off the packet header.

## <ins>Tile DMA packet config</ins>

Tile DMA can be configured so that data transfers are packetized through the use of the `AIE.dmaBdPacket` operation as shown below:
```
AIE.dmaBdPacket($packet_type, $packet_id)
```
An example of this inside a bd definition would be:
```
    ^bd0:
        AIE.useLock(%lock14_6, Acquire, 1)
        AIE.dmaBdPacket(0x4, 0xD) 
        AIE.dmaBd(<%buf14 : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock14_6, Release, 0)
        cf.br ^end
```
`$packet_type`: arbitary 3-bit value that is used to identify packet source

`$packet_id`: arbitary 5-bit value used to identify unique routes by switch boxes along the flow path

The configuration parameters that needs to match in order for packets to be successfully routed along a flow is:

(1) the packet id (as inserted by the tile DMA) 

(2) the switchbox configurations along the flow that those packets travel on

Once that are matching, the packetized data will be communicated the same as in the circuit switch case. 

The next question then is what benefit does packet switch mode offer then if it functions the same as circuit switch mode. And the answer is that we can acutally share the same physical routing with multiple packet flows such that data is shared over time on the same physical stream. We will explore that in more detail in [tutorial-7](../tutorial-7).


## <ins>Tutorial 6 Lab </ins>

1. Read through the [aie.mlir](aie.mlir) design. What is the packet ID used in this example? <img src="../images/answer1.jpg" title="0xD" height=25>

2. Measure the latency of the data transfer in packet mode and comapre it to the circuit switch mode result from [tutorial-4](../tutorial-4). How does it compare? <img src="../images/answer1.jpg" title="???" height=25>