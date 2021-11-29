# Benchmarks
This section provides example benchmark tests for measuring various aspects of the AIE, including data transfer, fill rate, and calibration measurements.
# Measurement Tools
## Performance Counters

Most of the benchmarks use performance counters for measurements. The performance counters can be used by specifying a start event, a stop event, and a reset event. The performance counter will trigger when the start event occurs, stop counting when the stop event occurs, and reset when the reset event occurs. We usually tie the performance counters to a lock acquire/lock release in memory so that we can time how long it takes for data to transfer.

## Program Counters
Program counters take in a start address of the assembly instruction and the stop address of the assembly instruction and measure the number of cycles between those two instructions.

## Timers
We can read the timer register to obtain the current timer value of an AI engine.
  
# Benchmark Tests

## Fill Rate Tests

  

These tests consist of benchmarks that measure the rate of data transfer across the AI Engine. They use performance counters in order to perform the measurements.

  

Tests 1, 2, 3, and 4 show different fill-rate tests.

  

## Core Measurements

  

These tests consist of benchmarks that measure operations in the core. They use performance counters in order to perform the measurements.



Tests 5, 6, 7, and 8 show different core measurements.

  

## Calibration Tests

  

These tests measure various calibration measurements of the broadcast and stream delay. They measure how long the broadcast signal takes to travel, as well as the stream delay when sending data across tiles. They use performance counters in order to perform the measurements.


Tests 9, 10, 11, and 12 show different calibration measurements.

  

## Other Measurement Examples


Test 13 shows the use of program counters for measuring operations in the AIE core.
  
Test 14 shows the use of timers, which can be used in order to measure the current timer value in an AIE tile.


## Benchmark Results on the VCK190

| Benchmark | Description                                                                                       | Result (cycles) @ 1GHz                     |
|-----------|---------------------------------------------------------------------------------------------------|--------------------------------------------|
| 01        | Measures the data transfer speed from the DDR to Local Memory in a tile                           | For 4096x4 bytes of data: 4437 ± 153.8     |
| 02        | Measures the data transfer speed from the Local Memory in a tile to the DDR                       | For 4096x4 bytes of data: 4096 ± 4148      |
| 03        | Measures the data transfer speed for 16 parallel DDR to Local Memory transfers                    | For 7168x4 bytes of data:  29389 ± 2984    |
| 04        | Measures the data transfer speed from a source tile local memory to destination tile local memory | 530                                        |
| 05        | Measures the cycles it takes a core to initialize                                                 | 52                                         |
| 06        | Measures the cycles it takes for a store operation in the AIE core                                | 57 (including initialization)              |
| 07        | Measures the cycles it takes for a lock acquire operation in the AIE core                         | 57 (including initialization)              |
| 08        | Measures the cycles it takes for a lock release operation in the AIE core                         | 57 (including initialization)              |
| 09        | Measures the cycles it take for the Shim to broadcast to other shim tiles                         | 4                                          |
| 10        | Measures the cycles it takes for a tile to broadcast horizontally (Each AIE tile has a core and memory module, with 16 broadcast wires horizontally and 32 vertically. Broadcast signals horizontally need to pass through both modules to travel to the next tile)                                | 2 per core/memory module                   |
| 11        | Measures the cycles it takes for a tile to broadcast vertically                                   | 2 per tile                                 |
| 12        | Measures the delay of transferring data on the stream                                             | 2 per node (North, South, East, West)      |