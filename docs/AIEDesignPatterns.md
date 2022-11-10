﻿
# AIE Basic Design Patterns

This document is an introduction to using the AIE dialect in practice and provides basic patterns that one would use in order to generate low level configurations for the AI engine. 

## Using AIE Cores

[Core Example](https://github.com/Xilinx/mlir-aie/tree/main/test/unit_tests/03_sync_with_locks/aie.mlir)

We can use the AIE Cores as below to perform some operations

Define a tile and a buffer
```
%tile13 = AIE.tile(1, 3)
%buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
```

Perform some operations on the buffer in the core
```
%core13 = AIE.core(%tile13) {
	%val1 = constant 7 : i32
	%idx1 = constant 3 : index
	%2 = addi %val1, %val1 : i32
	memref.store %2, %buf13_0[%idx1] : memref<256xi32>

	AIE.end
}


```

## Single-buffered Communication
[Single-buffer DMA example](https://github.com/Xilinx/mlir-aie/tree/main/test/unit_tests/05_tiledma/aie.mlir)

Define the AIE tiles you want to communicate between. Here Tile (7,1) will be the source and (7,2) the destination.

```
%t71 = AIE.tile(7, 1) // (Column, Row)
%t72 = AIE.tile(7, 2)
```
Set up switchboxes to connect the stream to DMA
```
%sw71 = AIE.switchbox(%t71) {
	AIE.connect<"DMA" : 0, "North" : 3>
}
%sw72 = AIE.switchbox(%t72) {
	AIE.connect<"South" : 3, "DMA" : 0>
}
```
Define the locks and buffers
```
%lock71 = AIE.lock(%t71, 0)  // Tile, Lock Number (0-15)
%lock72 = AIE.lock(%t72, 0) 

%buf71 = AIE.buffer(%t71) { sym_name = "a71" } : memref<512xi32>
%buf72 = AIE.buffer(%t72) { sym_name = "a72" } : memref<512xi32>
```

Start the Memory Map to Stream DMA from the source:
```
%mem71 = AIE.mem(%tile71) {
	%dma0 = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
	^bd0:
		AIE.useLock(%lock71, "Acquire", 0) // Acquire in State 0
		AIE.dmaBd(<%buf71 : memref<512xi32>, 0, 512>, 0)
		AIE.useLock(%lock71, "Release", 1) // Release in State 1
		br ^end 
	^end:
	AIE.end
}
```
Start the Stream to Memory Map DMA from the destination:

```
%mem72 = AIE.mem(%tile72) {
	%dma0 = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
	^bd0:
		AIE.useLock(%lock72, "Acquire", 0)
		AIE.dmaBd(<%buf72 : memref<512xi32>, 0, 512>, 0)
		AIE.useLock(%lock72, "Release", 1)
		br ^end 
	^end:
	AIE.end
}
```

We can also perform some operations in the AIE core using the same locks. When the locks are released in a certain state by the memory after the stream has finished transferring, we can acquire those locks in the core.


```
%c72 = AIE.core(%t72) {
	%val1 = constant 7 : i32
	%idx1 = constant 3 : index
	%2 = addi %val1, %val1 : i32
	
	AIE.useLock(%lock72, "Acquire", 1) // acquire for consume in the core
	memref.store %2, %buf72[%idx1] : memref<512xi32> //Store operation
	AIE.useLock(%lock72, "Release", 0) // release back to the memory
}
```
At the end, we release the lock back in state 0. This allows for the memory to re-acquire the lock in state 0.

## Double-buffered Communication

[Double-buffer DMA example](https://github.com/Xilinx/mlir-aie/tree/main/test/unit_tests/17_shim_dma_with_core/aie.mlir)

This example uses the same setup as the previous. For Tile (7,2) we can define an additional lock and buffer and change the buffers to be half the size:
```
%lock72_0 = AIE.lock(%t72, 0) 
%lock72_1 = AIE.lock(%t72, 1) 

%buf72_0 = AIE.buffer(%t72) { sym_name = "a72" } : memref<256xi32>
%buf72_1 = AIE.buffer(%t72) { sym_name = "b72" } : memref<256xi32>
```
Then we can write the Stream to Memory Map DMA transfer with 2 buffer descriptors:
```
%mem72 = AIE.mem(%t72) {
	%dma0 = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
	^bd0:
		AIE.useLock(%lock72_0, "Acquire", 0)
		AIE.dmaBd(<%buf72_0: memref<256xi32>, 0, 256>, 0)
		AIE.useLock(%lock72_0, "Release", 1)
		br ^bd1 // point to the next BD, or termination
	^bd1:
		AIE.useLock(%lock72_1, "Acquire", 0)
		AIE.dmaBd(<%buf72_1: memref<256xi32>, 0, 256>, 0)
		AIE.useLock(%lock72_1, "Release", 1)
		br ^bd0 // point to the next BD, or termination
	^end:

AIE.end

}
```

We can use the core in a similar fashion, using the two locks to perform operations on each buffer:
```
%c72 = AIE.core(%t72) {
	%val1 = constant 7 : i32
	%idx1 = constant 3 : index
	%idx2 = constant 10 : index

	%2 = addi %val1, %val1 : i32
	
	AIE.useLock(%lock72_0, "Acquire", 1) // acquire for consume in the core
	memref.store %2, %buf72[%idx1] : memref<512xi32> // store operation
	AIE.useLock(%lock72_0, "Release", 0) // release back to the memory
	
	AIE.useLock(%lock72_1, "Acquire", 1) // acquire for consume in the core
	memref.store %2, %buf72[%idx2] : memref<512xi32> // store operation
	AIE.useLock(%lock72_1, "Release", 0) // release back to the memory
}
```

## Controlling from the ARM Processor

[Controlling From ARM](https://github.com/Xilinx/mlir-aie/tree/main/test/unit_tests/17_shim_dma_with_core/aie.mlir)

We can perform some operations from the ARM processor and configure the lock to start the transfer. Here is a simple example where we write to a buffer, and begin the data transfer all from the host code.

We use a similar example to the single buffered communication:


```
%lock71 = AIE.lock(%t71, 0)  // Tile, Lock Number (0-15)
%lock72 = AIE.lock(%t72, 0) 

%buf71 = AIE.buffer(%t71) { sym_name = "a71" } : memref<512xi32>
%buf72 = AIE.buffer(%t72) { sym_name = "a72" } : memref<512xi32>
```

Start the Memory Map to Stream DMA from the source:
```
%mem71 = AIE.mem(%tile71) {
	%dma0 = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
	^bd0:
		AIE.useLock(%lock71, "Acquire", 1) // Acquire in State 0
		AIE.dmaBd(<%buf71 : memref<512xi32>, 0, 512>, 0)
		AIE.useLock(%lock71, "Release", 1) // Release in State 1
		br ^end 
	^end:
	AIE.end
}
```
Start the Stream to Memory Map DMA from the destination:

```
%mem72 = AIE.mem(%tile72) {
	%dma0 = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
	^bd0:
		AIE.useLock(%lock72, "Acquire", 0)
		AIE.dmaBd(<%buf72 : memref<512xi32>, 0, 512>, 0)
		AIE.useLock(%lock72, "Release", 1)
		br ^end 
	^end:
	AIE.end
}
```
Since %lock71 is now acquired at state 1, we need to manually release the lock into that state from the host side. This is because the default state of all locks are 0, so they are immediately able to be acquired.

In the host code, lets write to the buffer 71:
```
// We're going to stamp over the memory

for (int i = 0; i < DMA_COUNT; i++){
	mlir_aie_write_buffer_a71(ctx, i, 0xdeadbeef);
}
```

and release the lock:

```
XAieTile_LockRelease(&(TileInst[7][1]), 0, 1, 0); // Release lock
```

This allows the data transfer to begin

## Static DDR Configuration
[Static DDR](https://github.com/Xilinx/mlir-aie/tree/main/test/unit_tests/17_shim_dma_with_core/aie.mlir)

To read/write from DDR, we declare an external buffer with a location and size
```
%ext_buffer = AIE.external_buffer 0x02010004000 : memref<512 x i32>
```

We can then use the shimDMA to read/write from that location:

```
%lock70 = AIE.lock(%t70, 1)

%mem70 = AIE.mem(%tile70) {
	%dma0 = AIE.dmaStart("MM2S", 0, ^bd0, ^end) \\Read
	^bd0:
		AIE.useLock(%lock70 , "Acquire", 0)
		AIE.dmaBd(<%ext_buffer : memref<512xi32>, 0, 512>, 0)
		AIE.useLock(%lolock70 k72, "Release", 1)
		br ^end 
	^end:
	AIE.end
}
```

We can write to the external buffer using mmap in the host code:
```
#define BRAM_ADDR (0x4000+0x020100000000LL)
#define DMA_COUNT 512

int fd = open("/dev/mem", O_RDWR | O_SYNC);

if (fd != -1) {

	bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);

	for (int i=0; i<DMA_COUNT; i++) {
		bram_ptr[i] = 0xdeadbeef; //Write deadbeef
	}
}
```

## Dynamic DDR Configuration

In this pattern, we will show a design pattern for dynamic DDR configuration
```
module {

%t70 = AIE.tile(7, 0)
%t71 = AIE.tile(7, 1)
%t72 = AIE.tile(7, 2)

%buf72_0 = AIE.buffer(%t72) {sym_name="a"} : memref<256xi32>
%buf72_1 = AIE.buffer(%t72) {sym_name="b"} : memref<256xi32>

%l72_0 = AIE.lock(%t72, 0)
%l72_1 = AIE.lock(%t72, 1)

%m72 = AIE.mem(%t72) {

	%srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
	^bd0:
		AIE.useLock(%l72_0, "Acquire", 1)
		AIE.dmaBd(<%buf72_0 : memref<256xi32>, 0, 256>, 0)
		AIE.useLock(%l72_0, "Release", 0)
	br ^bd1
	^bd1:
		AIE.useLock(%l72_1, "Acquire", 1)
		AIE.dmaBd(<%buf72_1 : memref<256xi32>, 0, 256>, 0)
		AIE.useLock(%l72_1, "Release", 0)
	br ^bd0
	^end:

	AIE.end
}

AIE.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)

}
```

In our host code, we can create the external buffer and write to it:

```
#define BRAM_ADDR (0x4000+0x020100000000LL)
#define DMA_COUNT 512

int fd = open("/dev/mem", O_RDWR | O_SYNC);

if (fd != -1) {

	bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);

	for (int i=0; i<DMA_COUNT; i++) {
		bram_ptr[i] = 0xdeadbeef; //Write deadbeef
	}
}
```

We can write to buffer a and program the SHIM DMA using the XAIE API:

```
// Populate buffer with some data. It will get pushed into a stream connected
// to the ShimDMA.

for (int i=0; i<DMA_COUNT; i++) {
	uint32_t d = i+1;
	mlir_aie_write_buffer_a(ctx, i, d);
}

// Program the ShimDMA to write from stream to memory

auto burstlen = 4;
XAieDma_ShimInitialize(&(TileInst[7][0]), &ShimDmaInst1);
XAieDma_ShimBdSetAddr(&ShimDmaInst1, 1, HIGH_ADDR((u64)BRAM_ADDR), LOW_ADDR((u64)BRAM_ADDR), sizeof(u32) * DMA_COUNT);
XAieDma_ShimBdSetAxi(&ShimDmaInst1, 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
XAieDma_ShimBdWrite(&ShimDmaInst1, 1);
XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, 1); //Start the Buffer Descriptor
XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

```
We can then release the locks manually from the host code in order to begin the transfer:
```
XAieTile_LockRelease(&(TileInst[7][2]), 0, 0x1, 0);
XAieTile_LockRelease(&(TileInst[7][2]), 1, 0x1, 0);
```

## Using AIE ObjectFIFOs

[ObjectFIFO Example](https://github.com/Xilinx/mlir-aie/tree/main/test/objectFifo-stateful-transform/non_adjacency_test_1.aie.mlir)

An objectFIFO can be established between two or more tiles. Broadcast is possible from one producer tile to multiple consumer tiles.
Unlike a typical FIFO, elements are not pushed to nor popped from the objectFIFO. Instead, a pool of memory elements is allocated to the objectFIFO. 
Processes can then write to and read from these memory elements after acquiring them.

Define two tiles and create an AIE.objectFifo of depth two between them, with the two elements being of type <memref<16xi32>>:
```
%tile12 = AIE.tile(1, 2)
%tile33 = AIE.tile(3, 3)
%objFifo = AIE.objectFifo.createObjectFifo(%tile12, {%tile33}, 2) : !AIE.objectFifo<memref<16xi32>>
```
After subsequent conversion passes, each of the objectFifo elements is instantiated as an AIE.buffer with an AIE.lock.

objectFIFO operations have a 'port' attribute which indicates whether a tile is a 'producer' or a 'consumer' of that objectFIFO.
Operations can be performed on the objectFIFO in the cores: elements can be acquired from the objectFIFO and accessed via an AIE.objectFifoSubview type, then released: 
```
%core12 = AIE.core(%tile12) {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%height = arith.constant 12 : index

	scf.for %indexInHeight = %c0 to %height step %c1 {
		%subview = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
		%elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
		call @some_work(%elem0) : (memref<16xi32>) -> ()
		AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
	}
	
	AIE.end
}

%core33 = AIE.core(%tile33) {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%height = arith.constant 12 : index

	scf.for %indexInHeight = %c0 to %height step %c1 { 
		%subview = AIE.objectFifo.acquire<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
		%elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
		call @some_work(%elem0) : (memref<16xi32>) -> ()
		AIE.objectFifo.release<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
	}
	
	AIE.end
}
```

For correct execution, loops that contain objectFIFO operations must be unrolled based on objectFIFO size; the previous code in core12 becomes:
```
%core12 = AIE.core(%tile12) {
	%c0 = arith.constant 0 : index
	%c2 = arith.constant 2 : index
	%height = arith.constant 12 : index

	scf.for %indexInHeight = %c0 to %height step %c2 {
		%subview0 = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
		%elem00 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
		call @some_work(%elem00) : (memref<16xi32>) -> ()
		AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)

		%subview1 = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
		%elem10 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
		call @some_work(%elem10) : (memref<16xi32>) -> ()
		AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
	}
	
	AIE.end
}
```

At a higher abstraction level, a process can be registered to an objectFIFO using access patterns and work functions:
```
module @objectFIFO  {
    %tile12 = AIE.tile(1, 2)
    %tile33 = AIE.tile(3, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, {%tile33}, 2) : !AIE.objectFifo<memref<16xi32>>

    %prodAcqPattern = arith.constant dense<[1]> : tensor<1xi32>
    %prodRelPattern = arith.constant dense<[1]> : tensor<1xi32>
    %prodLength = arith.constant 12 : index
    func @producer_work() -> () {
        return
    }

    AIE.objectFifo.registerProcess<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, %prodAcqPattern : tensor<1xi32>, %prodRelPattern : tensor<1xi32>, @producer_work, %prodLength)
}
```

## Using AIE broadcast_packet

[broadcast_packet Example](https://github.com/Xilinx/mlir-aie/tree/main/test/unit_tests/23_broadcast_packet/aie.mlir)

The broadcast_packet operation is a logical connection that combines broadcast and packet-switch data transferring mechanism.

In this operation, the data streams with different packet-IDs will time-multiplexed use the single source port to broadcast 
data to multiple destinations.

The following example shows that two streams of data with different packet-ID (0x0 and 0x1) will time-multiplexed share the same 
source port (%t72, "DMA" : 0) to broadcast data to %t73, %t63(ID: 0x0) and %t74, %t64(ID: 0x1).

Define tiles
```
%t72 = AIE.tile(7, 2)
%t63 = AIE.tile(6, 3)
%t64 = AIE.tile(6, 4)
%t73 = AIE.tile(7, 3)
%t74 = AIE.tile(7, 4)

```

broadcast_packet 
```
AIE.broadcast_packet(%t72, "DMA" : 0){
  AIE.bp_id(0x0){
    AIE.bp_dest<%t73, "DMA" : 0>
    AIE.bp_dest<%t63, "DMA" : 0>
  }
  AIE.bp_id(0x1){
    AIE.bp_dest<%t74, "DMA" : 0>
    AIE.bp_dest<%t64, "DMA" : 0>
  }
}

```
