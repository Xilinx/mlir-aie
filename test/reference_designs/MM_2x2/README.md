## MM_2x2 Design example

### Overall Description<br>
This is an end-to-end matrix multiply example with size (32 * 64) * (64 * 64) by using the broadcast_packet data transferring mechanism.<br>

### Computation kernel<br>
&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; LHS &emsp; &nbsp; &nbsp; RHS &emsp; &emsp; Acc &emsp; &emsp; Output<br>
1. This design uses 4 AIEs with each AIE computes (32 * 32) * (32 * 32) + (32 * 32) = (32 * 32). In the single kernel design(kernel.cc) left-hand-side(LHS), right-hand-side(RHS), accumulator(Acc) and output matrix are all column based.<br>

### Mapping strategy<br>
1. The following figure shows the mapping strategy of this design in which LHS and Output each has two (32 * 32) tiles, RHS has four (32 * 32) tiles.<br>
![image](https://user-images.githubusercontent.com/77606152/182739157-8b34291b-7c7b-4796-a27a-907dcb0eca07.png)


### AIE array layout and data communication<br>
1. The communication from tile (6,0) MM2S0 channel is a broadcast_packet in which tile0 of LHS can be broadcast to tile(6,3) and tile (7,3). The data in tile1 of LHS can be broadcast to tile (6,4) and tile (7,4) by time-multiplexedly using the same dma channel with different packet ID.<br> 
2. One thing to notify is that the accumulator matrices are set to zero in the local memory of the AIEs in the third row. Thus, there isn't the dma channel to send the accumulator matrix to tile (6,3) and tile (7,3)<br>
![image](https://user-images.githubusercontent.com/77606152/182739011-d27f9e43-7468-43b5-bbfe-1ed399bfb2c6.png)

