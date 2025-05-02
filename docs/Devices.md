# Device support

This repo supports a number of devices, at some level of completeness.
In the below diagrams:
```C``` represents a 'core tile' including a CPU core, stream switch, local memories and DMAs
```M``` represents a 'memory tile' including a stream switch, larger local memories and DMAs
```P``` represents a 'Shim PL tile' including a stream connection to programmable logic.
```D``` represents a 'Shim DMA tile' including a Shim DMA, stream switch, and (in some devices), a stream connection to programmable logic.
```.``` represents an unusable tile.

## AIE Version 1

```aie.device(xcvc1902) { ... }```
This device is present on the VCK190 and VCK5000 boards.
50 Columns and 9 Rows
```
8 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
7 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
6 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
5 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
4 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
3 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
2 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
1 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
0 PPDDPPDDPPDDPPPPPPDDPPPPPPDDPPPPPPDDPPPPPPDDPPDDPP
  01234567890123456789012345678901234567890123456789
            1111111111222222222233333333334444444444
```

## AIE Version 2

```aie.device(xcve2302) {}```
17 Columns and 4 Rows
```
3 CCCCCCCCCCCCCCCCC
2 CCCCCCCCCCCCCCCCC
1 MMMMMMMMMMMMMMMMM
0 PPDDPPDDPPDDPPPPP
  01234567890123456
            1111111
```

```aie.device(xcve2802) {}```
This device is present on the V70 board
38 Columns and 11 Rows
```
0 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
9 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
8 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
7 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
6 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
5 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
4 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
3 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
2 MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
1 MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
0 PPDDPPDDPPPPPPDDPPPPPPDDPPPPPPDDPPDDPP
  01234567890123456789012345678901234567
            1111111111222222222233333333
```

```aie.device(npu1) {}```
This device is present in Ryzen Phoenix (e.g. 7940HS) and HawkPoint (e.g., 8040HS) SOCs.
5 Columns and 6 Rows
```
5 CCCCC
4 CCCCC
3 CCCCC
2 CCCCC
1 MMMMM
0 PDDDD
  01234
```

```aie.device(npu1_1col) {}```
```aie.device(npu1_2col) {}```
```aie.device(npu1_3col) {}```
```aie.device(npu1_4col) {}```
These devices represent a physical partition of an npu1 device, including a number of columns with a DMA shim tile.
N Columns and 6 Rows
```
5 CCCC
4 CCCC
3 CCCC
2 CCCC
1 MMMM
0 PDDD
  0..N
```

```aie.device(npu2) {}```
This NPU device is present in Ryzen AI: Strix, Strix Halo and Krackan Point SOCs.
8 Columns and 6 Rows
```
5 CCCCCCCC
4 CCCCCCCC
3 CCCCCCCC
2 CCCCCCCC
1 MMMMMMMM
0 DDDDDDDD
  01234567
```

```aie.device(npu2_1col) {}```
```aie.device(npu2_2col) {}```
```aie.device(npu2_3col) {}```
```aie.device(npu2_4col) {}```
```aie.device(npu2_5col) {}```
```aie.device(npu2_6col) {}```
```aie.device(npu2_7col) {}```
These devices represent a physical partition of an npu2 device, including a number of columns with a DMA shim tile.
N Columns and 6 Rows
```
5 CCCCCCC
4 CCCCCCC
3 CCCCCCC
2 CCCCCCC
1 MMMMMMM
0 DDDDDDD
  0.....N
```

-----

<p align="center">Copyright&copy; 2024 AMD/Xilinx</p>
