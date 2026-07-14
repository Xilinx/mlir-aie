<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# Device support

This repo supports a number of devices, at some level of completeness.
In the below diagrams:
```C``` represents a 'compute tile' including a CPU core, stream switch, local memories and DMAs
```M``` represents a 'mem tile' including a stream switch, larger local memories and DMAs
```P``` represents a 'Shim PL tile' including a stream connection to programmable logic.
```D``` represents a 'Shim DMA tile' including a Shim DMA, stream switch, and (in some devices), a stream connection to programmable logic.
```.``` represents an unusable tile.

```aie.device(npu1) {}```
This device is present in Ryzen Phoenix (e.g. 7940HS) and HawkPoint (e.g., 8040HS) SOCs.
4 Columns<sup>1</sup> and 6 Rows
```
5 CCCC
4 CCCC
3 CCCC
2 CCCC
1 MMMM
0 DDDD
  1234
```
> <sup>1</sup> The hidden zeroth-column of Phoenix NPUs is irregular and no longer exposed through MLIR-AIE.

```aie.device(npu1) {}```
```aie.device(npu1_1col) {}```
```aie.device(npu1_2col) {}```
```aie.device(npu1_3col) {}```
These devices represent a npu1 device, or a physical partition thereof, including a number of columns with a DMA shim tile.
N Columns and 6 Rows
```
5 CCCC
4 CCCC
3 CCCC
2 CCCC
1 MMMM
0 DDDD
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

<p align="center">Copyright&copy; 2024-2026 Advanced Micro Devices, Inc.</p>
