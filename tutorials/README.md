
# MLIR-based AI Engine Design Tutorial 

While it has been stated that the MLIR-based AI Engine toolchain is not intended to represent end-to-end compilation flows or be particularly easy to use for system design, the benefit of building systems using this representation is still very valuable and to that end, this design tutorial will help guide someone new to MLIR through the steps of building increasingly complex multi-core designs. In order to understand this MLIR-based representation for AI Engine design, it is important to first understand AI Engines and its architecture. 

The structure of this series of tutorials are as follows:
* Basic AI Engine architecture
* Tutorial 1 - modules, tile, buffer, core, kernel
* Tutorial 2 - single kernel compilation and simulation
* Tutorial 3 - communication (local memory), locks 
* Tutorial 4 - communication (tile DMA, switchbox?, logical routing)
* Tutorial 5 - communication (cascade)
* Tutorial 6 - communication (packet routing)
* Tutorial 7 - communication (broadcast)
* Tutorial 8 - communication (shim DMA, external memory aka DDR)
* Tutorial 9 - mlir-aie commands and utilities
* Example Design #1 - 2x2 Matrix Multiplication (Object FIFO)
* Example Design #2 - iDCT

## Basic AI Engine architecture
AI Engines are architected as 2D arrays consisting of multiple AI Engine tiles and allow for a very scalable solution across the Versal portfolio, ranging from 10s to 100s of AI Engines in a single device, servicing the compute needs of a breadth of applications.

To maximally utilize the full power of the AI Engine, designers are encouraged to familiarize themselves with  [ug1076](https://docs.xilinx.com/r/en-US/ug1076-ai-engine-environment/) for the AI Engine design environment and [am009](https://docs.xilinx.com/r/en-US/am009-versal-ai-engine) for detailed AI Engine functional specification.  For the purposes of this MLIR-based representation, we will focus primarily on the main AI Engine components and the communication between them. Single core programming and optimization, while an important aspect of AI Engine application development, will be described primarily as a means to facilitate data communication.

AI Engines are part of the larger Versal ACAP device and famiiarty with other CAP components such as the NoC, ARM processors, and custom PL components such as datamovers will help the design integrate their AI Engine design into a larger ACAP system design.

![heterogeneous-platform](https://www.xilinx.com/content/xilinx/en/products/technology/ai-engine/_jcr_content/root/imageTabParsys/childParsys-overview/xilinxcolumns_copy_c_1601636194/childParsys-1/xilinximage.img.png/1622238572946.png) 

Within the AI Engine region of the device, there is an array of AI Engine cores connected to one another through a number of communication structures (stream switches, local memories, and cascade streams). 
 
![](https://docs.xilinx.com/api/khub/maps/scNYG4asFKV~nqnjEkGwmA/resources/4QKgSQwqrYtSReOmABmdBw/content?Ft-Calling-App=ft%2Fturnkey-portal&Ft-Calling-App-Version=3.11.43)

And within an AI Engine tile, we see an ISA-based VLIW Vector processor with its own program memory and register file and its associated local data memory, which is shared with its immediate neighbors in a particular pattern (more on that later).
![](https://docs.xilinx.com/api/khub/maps/q_Yc6QkQHbaC2~Qz9NTtmg/resources/g8M48UDPavKcSu6HWN65FQ/content?Ft-Calling-App=ft%2Fturnkey-portal&Ft-Calling-App-Version=3.11.43)

### Communication
Focusing back on communication, there are 3 primary ways AI Engines communicate with one another: (1) local memory, (2) stream switch and (3) cascade. 

For local memory, each AI Engine is able to access the local memory of its immediate neighbor in all 4 cardinal directions. This tends to be the communication method with the most bandwidth as load and store units in AIE1 will access up to 256-bits per cycle.

>**Note for AIE1:** We have a notion of even and odd rows where the local memory of an AI Engine Core may be to the left (in even rows) or the right (in odd rows) of the AIE tile. As such, the local memory on the left for a given AIE tile may be its own local memory (for even rows) or that of its left neighbor (for odd rows). 

In the diagram below, we see data being communicated between AIE tiles through local memory in a pipelined or dataflow way.

![](https://docs.xilinx.com/api/khub/maps/scNYG4asFKV~nqnjEkGwmA/resources/46TJtyJx_RF00BGX0ErAXA/content?Ft-Calling-App=ft%2Fturnkey-portal&Ft-Calling-App-Version=3.11.43&filename=bzt1530655350975.image)

Second, we have the stream switch which moves data up to 32-bits per cycle. Here, data is moved through stream switches throughout the AI Engine array from a source AIE tile to destination one. These stream paths can be circuit switched or packet switched and can be managed via DMAs (TileDMA) or directly by the AIE core. This is the second most common method of data communication and is the only method for moving data between non-adjacent tiles and into/out of the AI Engine array. The diagram below shows a streaming mulitcast example where streams are multicast from one AIE tile to 3 destinations.
![](https://docs.xilinx.com/api/khub/maps/scNYG4asFKV~nqnjEkGwmA/resources/rJt9bOfzmlQdCPlCnG5_WQ/content?Ft-Calling-App=ft%2Fturnkey-portal&Ft-Calling-App-Version=3.11.43&filename=ixc1530655536100.image)

Finally, we have cascade streams which has the widest data width (384-bits per cycle) but only moves data between accumulator registers of horizontally adjacent neighbors (in AIE1). In addition for AIE1, the cascade direction is right-to-left for even rows and left-to-right for odd rows (wrapping around and up such that the rightmost tile in row1 has a cascade connection to the rightmost tile in row 2, while the leftmost tile of row 2 cascades into the leftmost tile of row 3).

There are fixed number of vertical and horizontal streams routed by the stream switch so balancing the data movement over these shared resources is an important part of efficient AI Engine design. 

We've only begun to touch the the processing and communication capabilities of the AI Engine so please refer to the online specification and user guides (as listed earlier) for more details.


