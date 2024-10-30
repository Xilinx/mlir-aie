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
	%dma0 = AIE.dma_start("MM2S", 0, ^bd0, ^end)
	^bd0:
		AIE.use_lock(%lock71, "Acquire", 0) // Acquire in State 0
		AIE.dma_bd(%buf71 : memref<512xi32>, 0, 512)
		AIE.use_lock(%lock71, "Release", 1) // Release in State 1
		br ^end 
	^end:
	AIE.end
}
```
Start the Stream to Memory Map DMA from the destination:

```
%mem72 = AIE.mem(%tile72) {
	%dma0 = AIE.dma_start("S2MM", 0, ^bd0, ^end)
	^bd0:
		AIE.use_lock(%lock72, "Acquire", 0)
		AIE.dma_bd(%buf72 : memref<512xi32>, 0, 512)
		AIE.use_lock(%lock72, "Release", 1)
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
	
	AIE.use_lock(%lock72, "Acquire", 1) // acquire for consume in the core
	memref.store %2, %buf72[%idx1] : memref<512xi32> //Store operation
	AIE.use_lock(%lock72, "Release", 0) // release back to the memory
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
	%dma0 = AIE.dma_start("S2MM", 0, ^bd0, ^end)
	^bd0:
		AIE.use_lock(%lock72_0, "Acquire", 0)
		AIE.dma_bd(%buf72_0: memref<256xi32>, 0, 256)
		AIE.use_lock(%lock72_0, "Release", 1)
		br ^bd1 // point to the next BD, or termination
	^bd1:
		AIE.use_lock(%lock72_1, "Acquire", 0)
		AIE.dma_bd(%buf72_1: memref<256xi32>, 0, 256)
		AIE.use_lock(%lock72_1, "Release", 1)
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
	
	AIE.use_lock(%lock72_0, "Acquire", 1) // acquire for consume in the core
	memref.store %2, %buf72[%idx1] : memref<512xi32> // store operation
	AIE.use_lock(%lock72_0, "Release", 0) // release back to the memory
	
	AIE.use_lock(%lock72_1, "Acquire", 1) // acquire for consume in the core
	memref.store %2, %buf72[%idx2] : memref<512xi32> // store operation
	AIE.use_lock(%lock72_1, "Release", 0) // release back to the memory
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
	%dma0 = AIE.dma_start("MM2S", 0, ^bd0, ^end)
	^bd0:
		AIE.use_lock(%lock71, "Acquire", 1) // Acquire in State 0
		AIE.dma_bd(%buf71 : memref<512xi32>, 0, 512)
		AIE.use_lock(%lock71, "Release", 1) // Release in State 1
		br ^end 
	^end:
	AIE.end
}
```
Start the Stream to Memory Map DMA from the destination:

```
%mem72 = AIE.mem(%tile72) {
	%dma0 = AIE.dma_start("S2MM", 0, ^bd0, ^end)
	^bd0:
		AIE.use_lock(%lock72, "Acquire", 0)
		AIE.dma_bd(%buf72 : memref<512xi32>, 0, 512)
		AIE.use_lock(%lock72, "Release", 1)
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

We can then use the shim_dma to read/write from that location:

```
%lock70 = AIE.lock(%t70, 1)

%mem70 = AIE.mem(%tile70) {
	%dma0 = AIE.dma_start("MM2S", 0, ^bd0, ^end) \\Read
	^bd0:
		AIE.use_lock(%lock70 , "Acquire", 0)
		AIE.dma_bd(%ext_buffer : memref<512xi32>, 0, 512)
		AIE.use_lock(%lolock70 k72, "Release", 1)
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

	%srcDma = AIE.dma_start("MM2S", 0, ^bd0, ^end)
	^bd0:
		AIE.use_lock(%l72_0, "Acquire", 1)
		AIE.dma_bd(%buf72_0 : memref<256xi32>, 0, 256)
		AIE.use_lock(%l72_0, "Release", 0)
	br ^bd1
	^bd1:
		AIE.use_lock(%l72_1, "Acquire", 1)
		AIE.dma_bd(%buf72_1 : memref<256xi32>, 0, 256)
		AIE.use_lock(%l72_1, "Release", 0)
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
Unlike a typical FIFO, elements are not pushed to nor popped from the objectFIFO. Instead, a pool of memory elements is allocated to the objectFIFO by the objectFIFO lowering pass, i.e., AIEObjectFifoStatefulTransform.mlir. 

Processes can then write to and read from these memory elements after acquiring them.

Define two tiles and create an AIE.objectfifo named @of0 of depth two between them, with the two elements being of type <memref<16xi32>>:
```
%tile12 = AIE.tile(1, 2)
%tile33 = AIE.tile(3, 3)
AIE.objectfifo @of0 (%tile12, {tile33}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>
```
After subsequent conversion passes, each of the objectFifo elements is instantiated as an AIE.buffer with an AIE.lock.

objectFIFO operations have a 'port' attribute which indicates whether a tile is a 'producer' or a 'consumer' of that objectFIFO.
Operations can be performed on the objectFIFO in the cores: elements can be acquired from the objectFIFO and accessed via an AIE.objectfifosubview type, then released: 
```
%core12 = AIE.core(%tile12) {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%height = arith.constant 12 : index

	scf.for %indexInHeight = %c0 to %height step %c1 {
		%subview = AIE.objectfifo.acquire @of0 (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
		%elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
		call @some_work(%elem0) : (memref<16xi32>) -> ()
		AIE.objectfifo.release @of0 (Produce, 1)
	}
	
	AIE.end
}

%core33 = AIE.core(%tile33) {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%height = arith.constant 12 : index

	scf.for %indexInHeight = %c0 to %height step %c1 { 
		%subview = AIE.objectfifo.acquire @of0 (Consume, 1) : !AIE.objectfifosubview<memref<16xi32>>
		%elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
		call @some_work(%elem0) : (memref<16xi32>) -> ()
		AIE.objectfifo.release @of0 (Consume, 1)
	}
	
	AIE.end
}
```

For correct execution, objectfifo operations must be lowered such that each iteration of execution, new elements are accessed (based on acquire / release patterns). Two different lowering techniques are described below:

In the default lowering, loops that contain objectFIFO operations are unrolled based on objectFIFO size; the previous code in core12 becomes:
```
%core12 = AIE.core(%tile12) {
	%c0 = arith.constant 0 : index
	%c2 = arith.constant 2 : index
	%height = arith.constant 12 : index

	scf.for %indexInHeight = %c0 to %height step %c2 {
		%subview0 = AIE.objectfifo.acquire @of0 (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
		%elem00 = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
		call @some_work(%elem00) : (memref<16xi32>) -> ()
		AIE.objectfifo.release @of0 (Produce, 1)

		%subview1 = AIE.objectfifo.acquire @of0 (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
		%elem10 = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
		call @some_work(%elem10) : (memref<16xi32>) -> ()
		AIE.objectfifo.release @of0 (Produce, 1)
	}
	
	AIE.end
}
```

Another lowering technique generates MLIR operations that ensure the acquire / release patterns are taken into account at runtime and their effects are stored in a global buffer. This global state buffer is then used to correctly access objectfifos using a SCF.IndexSwitchOps; the previous code in core12 becomes:
```
%of0_buff_0 = aie.buffer(%tile_0_2) {sym_name = "of0_buff_0"} : memref<16xi32> 
%of0_buff_1 = aie.buffer(%tile_0_2) {sym_name = "of0_buff_1"} : memref<16xi32> 
%of0_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "of0_prod_lock"}
%of0_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
%buffer_0_2 = aie.buffer(%tile_0_2) : memref<1xindex> 
%core_0_2 = aie.core(%tile_0_2) {
	%c0 = arith.constant 0 : index
	%c0_0 = arith.constant 0 : index
	%c2 = arith.constant 2 : index
	memref.store %c0, %buffer_0_2[%c0_0] : memref<1xindex>
	%c0_1 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%c12 = arith.constant 12 : index
	scf.for %arg0 = %c0_1 to %c12 step %c1 {
		aie.use_lock(%of0_prod_lock, AcquireGreaterEqual, 1)
		%0 = memref.load %buffer_0_2[%c0_0] : memref<1xindex>
		%1 = scf.index_switch %0 -> memref<16xi32> 
		case 0 {
			scf.yield %of0_buff_0 : memref<16xi32>
		}
		case 1 {
			scf.yield %of0_buff_1 : memref<16xi32>
		}
		default {
			scf.yield %of0_buff_0 : memref<16xi32>
		}
		func.call @some_work(%1) : (memref<16xi32>) -> ()
		aie.use_lock(%of0_cons_lock, Release, 1)
		%2 = memref.load %buffer_0_2[%c0_0] : memref<1xindex>
		%c1_2 = arith.constant 1 : index
		%3 = arith.addi %2, %c1_2 : index
		%4 = arith.remsi %3, %c2 : index
		memref.store %4, %buffer_0_2[%c0_0] : memref<1xindex>
	}
	aie.end
}
```
This lowering can be enabled for each core by setting the `dynamic_objfifo_lowering` attribute of the CoreOp to true, or enabled for all the cores in the design at once by setting the `dynamic-objFifos` flag of aiecc (which is then passed to the --aie-objectFifo-stateful-transform lowering pass).

ObjectFIFOs can be established between tiles on the shim row and AIE tiles in order to bring data in from or out to external memory locations. These external memory locations are pointed to using AIE.external_buffer operations and they need to be explicitly registered to an objectFIFO so that it knows where the data has been allocated externally (in this case, the objectFIFO lowering will only allocate memory elements required by AIE tiles):
```
module @objectFIFO  {
    %tile10 = AIE.tile(1, 0)
    %tile33 = AIE.tile(3, 3)

    AIE.objectfifo @of1 (%tile10, {tile33}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>

    %ext_buffer_in_0 = AIE.external_buffer {sym_name = "ext_buffer_in_0"}: memref<64xi32>
    %ext_buffer_in_1 = AIE.external_buffer {sym_name = "ext_buffer_in_1"}: memref<64xi32>
    AIE.objectfifo.register_external_buffers @of1 (%tile10, { %ext_buffer_in_0, %ext_buffer_in_1 }) : (memref<64xi32>, memref<64xi32>)
}
```

It is possible to copy data from one objectFifo to another. This copy can be done explicitly within the AIE cores, or implicitly using the tile DMAs. The latter case is not as much a copy as it is re-using the same memory buffers when receiving data on an input channel and sending the data out on an output channel. At the objectFIFO abstraction, this is called 'linking' two objectFIFOs. It is most commonly done inside of Mem tiles which have more memory than AIE tiles. 
```
module @objectFIFO  {
    %tile20 = AIE.tile(2, 0)
    %tile22 = AIE.tile(2, 2)
    %tile24 = AIE.tile(2, 4)

    AIE.objectfifo @of1 (%tile20, { %tile22 }, 2 : i32) : !AIE.objectfifo<memref<16xi32>>
	AIE.objectfifo @of2 (%tile22, { %tile24 }, 2 : i32) : !AIE.objectfifo<memref<16xi32>>

	AIE.objectfifo.link [@of1] -> [@of2] ()
}
```

At a higher abstraction level, a process can be registered to an objectFIFO using access patterns and work functions:
```
module @objectFIFO  {
    %tile12 = AIE.tile(1, 2)
    %tile33 = AIE.tile(3, 3)

    AIE.objectfifo @of1 (%tile12, {tile33}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>

    %prodAcqPattern = arith.constant dense<[1]> : tensor<1xi32>
    %prodRelPattern = arith.constant dense<[1]> : tensor<1xi32>
    %prodLength = arith.constant 12 : index
    func @producer_work() -> () {
        return
    }

    AIE.objectfifo.register_process @of1 (Produce, %prodAcqPattern : tensor<1xi32>, %prodRelPattern : tensor<1xi32>, @producer_work, %prodLength)
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
