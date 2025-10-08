<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 7 - Communication via switchboxes (broadcast)</ins>

At the switchbox level much like we did in the [Tutorial-4 - switchbox](../../tutorial-4/switchbox), for circuit switch broadcast communication, we can simply replace `AIE.flow` with `AIE.switchbox`. Where we actually more control over the switchbox behavior is in the packet switch case.

We already introduced the `AIE.switchbox` operator where circuit switch connections between ports can be made with the `AIE.connect` operator. For packet switch however, we use `AIE.amsel`, `AIE.masterset` and `AIE.packet_rules/ AIE.rule` to specify the packet switch behavior.

A sample conversion from the [flow](../flow) section which looked like this:
```
AIE.broadcast_packet(%tile14, DMA: 0) {
  AIE.bp_id(0xD) {
    AIE.bp_dest<%tile34, DMA: 1>
    AIE.bp_dest<%tile35, DMA: 1>
  }
}
```
would be lowered to something like this:
```
%0 = AIE.tile(1, 4)
%1 = AIE.switchbox(%0) {
  %20 = AIE.amsel<0> (0)
  %21 = AIE.masterset(East : 0, %20)
  AIE.packet_rules(DMA : 0) {
    AIE.rule(31, 13, %20)
  }
}
%18 = AIE.tile(2, 4)
%19 = AIE.switchbox(%18) {
  %20 = AIE.amsel<0> (0)
  %21 = AIE.masterset(East : 0, %20)
  AIE.packet_rules(West : 0) {
    AIE.rule(31, 13, %20)
  }
}
%2 = AIE.tile(3, 4)
%3 = AIE.switchbox(%2) {
  %20 = AIE.amsel<0> (0)
  %21 = AIE.masterset(DMA : 1, %20)
  %22 = AIE.masterset(North : 0, %20)
  AIE.packet_rules(West : 0) {
    AIE.rule(31, 13, %20)
  }
}
%4 = AIE.tile(3, 5)
%5 = AIE.switchbox(%4) {
  %20 = AIE.amsel<0> (0)
  %21 = AIE.masterset(DMA : 1, %20)
  AIE.packet_rules(South : 0) {
    AIE.rule(31, 13, %20)
  }
}
```
The packet switchbox has a lot of flexibility in how packets can be arbitrated and from any input port to any output port. Since there is a limit to the number of arbiters and master select, it is not the case that all packets with a given packet ID can be routed from any input to any output. But given the flexibility of the switchbox, there are multiple ways to configure it to get similar routing behavior. One way to think about the switchbox packet routing resources is as follows:

1. pool of input ports where packet rules are defined (associate packet ID/mask with an "arbiter/ master select (amsel)")
2. pool of "arbiters/ master select (amsel)" units that can be attached to >= 1 input port and >=1 output port
3. pool of output ports that can only have 1 arbiter attached to each output (though it can have >1 master sel with that same arbiter value)

The steps that a packet goes through when it arrives in an input port is:
1. Packets come in on input (slave) port and is sorted to "arbiters/ master select (amsel)" based on packet rules for that input port
2. "amsel" are associated with 1 or more output (master) port. If an arbiter has multiple input ports associated with it, then it will arbitrate between them via round robin. If an arbiter has multiple output ports, then it will broadcast to all ports simultaneously. 

> There are a lot of different ways these building blocks can be combined to create a particular set of packet routing rules with different performances. We do not hope to address all the nuances of this but it is a good area of future research and optimizations.

Packets that arrive in the switchbox are sorted into "arbiters/master select" based on packet rules on input (slave) switchbox ports. These rules are composed of a mask and value which the packet ID is matched against. An example of packet rules for an input port is:
```
AIE.packetRules("DMA" : 0) {
  AIE.rule(0x1F, 0x2, %amsel4_1)
  AIE.rule(0x1B, 0x1, %amsel3_2)
}
```
In this example, we define 2 rules for the input port "DMA" channel 0. The first is a 0x1F mask where a match to ID value 0x2 is sorted to arbiter/master select %amsel4_1 (defined elsewhere). The second rule for this input port is a 0x1B mask where a packet ID of 0x1 goes to arbiter/ master select %amsel4_2. Note that with a mask of 0x1B, a packet ID of 0x1 or 0x5 are both matches.

The previous block alludes to defining "arbiters/ master select" with `AIE.amsel`. These are then attached to a one or more output (master) ports via the `AIE.masterset` operation. This means a master output port can be associated with multiple amsel sets.

| Operators | Description |
|--------------------|-------------|
|`AIE.amsel<arbiter>(master select)` | Combination of arbiter (6 values) and master select (4 values). Generally speaking, an arbiter will use multiple master select (msel) values if an arbiter is connected to more than 1 output port and its not set to always broadcast to all of them. |
|`AIE.masterset (<master port>, <amsels>, ... <amsels>)`| Configures a master port to use the "arbiter and master select" defined by one or more `AIE.amsel`. A master port is the output ports for a switchbox which can include multiple master ports in all 4 directions, core, DMA, and fifo. There is generally just one `amsel` we can use but that can be more than one if each `amsel` has the same arbiter value (which is like 1 arbiter connected to >1 output ports). |

Example master ports
| Bundle | Channels (Out) |
|-------|---|
| DMA   | 2 |
| Core  | 2 |
| West  | 4 |
| East  | 4 |
| North | 6 |
| South | 4 |



## <ins>Tutorial 7 Lab </ins>

1. 