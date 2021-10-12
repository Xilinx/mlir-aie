﻿# AIE Flows and Routing

Introduction on how to connect tiles in the AIE physical dialect

## AIE Tile Routing

Define the AIE tiles you want to communicate between. Here Tile (7,1) will be the source and (7,2) the destination.

```
%t71 = AIE.tile(7, 1) // (Column, Row)
%t72 = AIE.tile(7, 2)
%t73 = AIE.tile(7, 3)
```
Set up a switchboxes to connect the DMA to the stream:
```
%sw71 = AIE.switchbox(%t71) {
	AIE.connect<"DMA" : 0, "North" : 1>
}
```
The AIE.connect function must exist inside an AIE.switchbox or AIE.shimmux operation. The numbers designate the source and destination channel number of the stream that we are trying to connect.

The switchbox in tile (7,1) connects the DMA channel 0 to North channel 1. If we define a switchbox in tile (7,2) as so:
 ```
%sw72 = AIE.switchbox(%t72) {
	AIE.connect<"South" : 1, "DMA" : 0>
}
```

The stream will automatically be connected, due to the fact that North 1 in tile (7,1) is connected to South 1 in tile (7,2). We can create multiple connections in the AIE switchbox:  

 ```
%sw71 = AIE.switchbox(%t71) {
	AIE.connect<"DMA" : 0, "North" : 0>
	AIE.connect<"North" : 2, "DMA" : 1>
	AIE.connect<"East" : 3, "West" : 2>
}
```

as long as we don't have duplicate destinations.

## AIE Shimmux 

Shim tiles are special tiles at row 0 of the AIE array. You can configure a switch in the Shim tile using 'shimmux' in addition to the switchbox operations:

 ```
%sw70 = AIE.shimmux(%t70) {
	AIE.connect<"North" : 2, "DMA" : 1>
}
```

Then, we can connect the DMA to the rest of the array like so:

 ```
%s70 = AIE.switchbox(%t70) {
	AIE.connect<"North" : 0, "South" : 2>
}
```

In order to read and write from the DDR using all available channels, a shimmux can be created like so:

```
%sw70 = AIE.shimmux(%t70) { 
  AIE.connect<"DMA" : 0, "North" : 3> \\ read
  AIE.connect<"DMA" : 1, "North" : 7> \\ read
  AIE.connect<"North" : 2, "DMA" : 0> \\ write
  AIE.connect<"North" : 3, "DMA" : 1> \\ write
}
```

The shimmux always connects to a switchbox to its north, located within the same tile. The shimmux connects the Shim DMA channels to specific stream channels: i.e. exiting the array (write), streams 2 and 3 from the switchbox connect to Shim DMA channels, but entering the array (read), streams 3 and 7 to the switchbox connect from Shim DMA channels. The shimmux is then connected to the switchbox to route the streams to/from the array as shown above. 

It is important to note how the shimmux is modeled in MLIR compared to libXAIEV1. While the shimmux is modeled in MLIR matching the convention of a switchbox, due to the fact that it exists in the same tile as the switchbox, the channel directions are inverted (North to South) when lowered to libXAIEV1. 

## AIE Flows

In order to connect larger distances, AIE flows exist so that we don't have to declare a switchbox for each tile.  Here we create a flow from the stream in tile (7,1) to the DMA in tile (7,3). Then, we create the flow another flow from the DMA in tile (7,3) to the stream in (7,1). We now don't have to define any switchbox in tile (7,2).

```
AIE.flow(%t71, "South" : 3, %t73, "DMA" : 0)
AIE.flow(%t73, "DMA" : 1, %t71, "South" : 2)
```
Flows can also be used to connect Shim DMAs to Tile DMAs over a distance: 

```
AIE.flow(%t70, "DMA" : 0, %t73, "DMA" : 0)
AIE.flow(%t73, "DMA" : 1, %t70, "DMA" : 0)
```

