# AIE Flows and Routing

<<<<<<< HEAD
<<<<<<< HEAD
Introduction on how to connect tiles in the AIE physical dialect
=======
Introduction on how connect tiles in the AIE physical dialect
>>>>>>> 2ca9878 (Updated File names)
=======
Introduction on how to connect tiles in the AIE physical dialect
>>>>>>> ed0e792 (Updated examples with unit test real code example, fixed typos)

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

The switchbox in tile (7,1) is connected to North channel 1. If we define a switchbox in tile (7,2) as so:
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

You can configure a switch in the PL Shim using Shimmux in addition to the switchbox operations. Shim tiles are special tiles at row 0 of the AIE array.

 ```
%sw70 = AIE.shimmux(%t70) {
	AIE.connect<"North" : 0, "DMA" : 1>
}
```

then we can connect the DMA to the rest of the array like so:

 ```
%s70 = AIE.switchbox(%t70) {
	AIE.connect<"DMA" : 1, "East" : 2>
}
```

In order to read and write from the DDR, a shimmux must be created like so:

```
%sw70 = AIE.shimmux(%t70) { AIE.connect<"DMA" : 0, "North" : 1> } \\ read
%sw70 = AIE.shimmux(%t70) { AIE.connect<"North" : 2, "DMA" : 1>} \\ write
```

The difference between these is from the south and entering the PL, streams 2 and 3 from the stream switch connect to the shim DMA, but from the north and entering the PL, streams 3 and 7 from the stream siwtch connect to the DMA.

## AIE Flows

In order to connect larger distances, AIE flows exist so that we don't have to declare a switchbox for each tile.  Here we create a flow from the stream in tile (7,1) to the DMA in tile (7,3). Then, we create the flow another flow from the DMA in tile (7,3) to the stream in (7,1). We now don't have to define any switchbox in tile (7,2).

```
AIE.flow(%t71, "South" : 3, %t73, "DMA" : 0)
AIE.flow(%t73, "DMA" : 1, %t71, "South" : 2)
```

